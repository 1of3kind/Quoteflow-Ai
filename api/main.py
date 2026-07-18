"""Main FastAPI application for QuoteFlow AI."""

import os
import logging
from typing import Optional, List
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status, Form, UploadFile, File, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.conversation_manager import ConversationManager, ConversationStage, get_response
from core.image_analyzer import get_analyzer
from core.quote_calculator import QuoteCalculator
from services.sms_handler import SMSHandler
from services.email_processor import EmailProcessor
from services.voice_handler import VoiceHandler
from services.notification_service import NotificationService
from workers.celery_tasks import process_photos_and_quote, daily_follow_ups

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

security = HTTPBearer()
API_KEY = os.getenv("FEEDBACK_API_KEY", "beta-secret-key-2026")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


class TradeSelect(BaseModel):
    customer_id: str
    trade: str
    phone: Optional[str] = None
    email: Optional[str] = None


class QuoteAccept(BaseModel):
    customer_id: str
    quote_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("QuoteFlow AI starting...")
    app.state.conversations = ConversationManager()
    app.state.sms = SMSHandler()
    app.state.email = EmailProcessor()
    app.state.voice = VoiceHandler()
    app.state.notifier = NotificationService()
    app.state.calculator = QuoteCalculator()
    yield
    logger.info("QuoteFlow AI shutting down...")


app = FastAPI(
    title="QuoteFlow AI",
    description="AI-powered quoting for small businesses. Customers send photos, AI analyzes and generates instant quotes.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "QuoteFlow AI",
        "version": "3.0.0",
        "description": "Send photos → AI analyzes → Instant quote via SMS/Email",
        "trades": ["landscaping", "roofing", "plumbing", "autobody", "electrical"],
        "endpoints": [
            "/health",
            "/webhook/sms",
            "/webhook/email",
            "/voice/welcome",
            "/voice/trade-select",
            "/quote/start",
            "/quote/accept",
            "/admin/conversations",
            "/admin/analytics",
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_conversations": app.state.conversations.get_active_count(),
    }


@app.post("/webhook/sms")
async def webhook_sms(request: Request):
    """Handle incoming SMS with photos from customers."""
    form = await request.form()
    data = app.state.sms.parse_inbound(dict(form))

    customer_id = f"sms_{data['from']}"
    body = data["body"].lower().strip()
    media_urls = [url for url in data["media_urls"] if url]

    conv = app.state.conversations.get_or_create(
        customer_id=customer_id,
        phone=data["from"],
    )

    trades = ["landscaping", "roofing", "plumbing", "autobody", "electrical"]
    if body in trades and conv.stage == ConversationStage.GREETING:
        conv.trade = body
        app.state.conversations.update_stage(customer_id, ConversationStage.TRADE_SELECT)

        from config.pricing_configs import get_trade_config
        config = get_trade_config(body)
        required = ", ".join(config.required_photos)

        response_text = get_response("trade_select", required_photos=required)
        return app.state.sms.create_response(response_text)

    if media_urls and conv.trade:
        conv.add_photos(media_urls)
        app.state.conversations.update_stage(customer_id, ConversationStage.PHOTO_RECEIVED)
        process_photos_and_quote.delay(customer_id, media_urls, conv.trade)

        response_text = get_response("analyzing")
        return app.state.sms.create_response(response_text)

    if body in ("accept", "yes", "book", "schedule") and conv.quote_id:
        app.state.conversations.update_stage(customer_id, ConversationStage.BOOKED)
        response_text = get_response("booked", quote_id=conv.quote_id, total="TBD")
        return app.state.sms.create_response(response_text)

    if conv.stage == ConversationStage.GREETING:
        response_text = get_response("greeting")
    else:
        response_text = "I'm not sure what you mean. Reply with your trade type or send photos of the work area."

    return app.state.sms.create_response(response_text)


@app.post("/webhook/email")
async def webhook_email(request: Request):
    """Handle incoming emails with photo attachments."""
    payload = await request.json()
    email = app.state.email.parse_inbound(payload)

    customer_id = f"email_{email.from_email}"
    conv = app.state.conversations.get_or_create(
        customer_id=customer_id,
        email=email.from_email,
    )

    trades = ["landscaping", "roofing", "plumbing", "autobody", "electrical"]
    body_lower = (email.subject + " " + email.body_text).lower()
    detected_trade = next((t for t in trades if t in body_lower), None)

    if detected_trade and not conv.trade:
        conv.trade = detected_trade
        app.state.conversations.update_stage(customer_id, ConversationStage.TRADE_SELECT)

    photo_urls = [att["url"] for att in email.attachments if "image" in att.get("content_type", "")]
    if photo_urls and conv.trade:
        conv.add_photos(photo_urls)
        process_photos_and_quote.delay(customer_id, photo_urls, conv.trade)

        await app.state.email.send_quote_email(
            email.from_email,
            "Photo received - analyzing now",
            "Thanks for the photos! Our AI is analyzing them now. You'll receive your quote within 2 minutes.",
            "ack",
        )

    return {"status": "processed"}


@app.post("/voice/welcome")
async def voice_welcome():
    """Twilio voice webhook: initial greeting."""
    return app.state.voice.create_welcome_response()


@app.post("/voice/trade-select")
async def voice_trade_select(Digits: str = Form(...)):
    """Handle trade selection from phone keypad."""
    trades = {
        "1": "landscaping",
        "2": "roofing",
        "3": "plumbing",
        "4": "autobody",
        "5": "electrical",
    }
    trade = trades.get(Digits, "unknown")
    return app.state.voice.create_photo_instructions(trade)


@app.post("/quote/start", dependencies=[Depends(verify_token)])
async def start_quote(data: TradeSelect):
    """Manually start a quote (for web dashboard)."""
    conv = app.state.conversations.get_or_create(
        customer_id=data.customer_id,
        phone=data.phone,
        email=data.email,
    )
    conv.trade = data.trade
    app.state.conversations.update_stage(data.customer_id, ConversationStage.TRADE_SELECT)

    from config.pricing_configs import get_trade_config
    config = get_trade_config(data.trade)

    return {
        "customer_id": data.customer_id,
        "trade": data.trade,
        "required_photos": config.required_photos,
        "message": f"Please send photos of: {', '.join(config.required_photos)}",
    }


@app.post("/quote/accept", dependencies=[Depends(verify_token)])
async def accept_quote(data: QuoteAccept):
    """Customer accepts a quote."""
    conv = app.state.conversations.get(data.customer_id)
    if not conv or conv.quote_id != data.quote_id:
        raise HTTPException(status_code=404, detail="Quote not found")

    app.state.conversations.update_stage(data.customer_id, ConversationStage.BOOKED)
    return {
        "status": "booked",
        "quote_id": data.quote_id,
        "customer_id": data.customer_id,
        "next_steps": "An agent will contact you within 24 hours to schedule.",
    }


@app.get("/admin/conversations", dependencies=[Depends(verify_token)])
async def list_conversations(stage: Optional[str] = None):
    """List all conversations with optional filter."""
    convs = app.state.conversations.conversations.values()
    if stage:
        convs = [c for c in convs if c.stage.value == stage]
    return {
        "count": len(convs),
        "conversations": [c.to_dict() for c in convs],
    }


@app.get("/admin/analytics", dependencies=[Depends(verify_token)])
async def analytics():
    """Get business analytics."""
    return {
        "total_conversations": len(app.state.conversations.conversations),
        "active": app.state.conversations.get_active_count(),
        "conversion_rate": round(app.state.conversations.get_conversion_rate(), 2),
        "by_stage": {
            stage.value: len([
                c for c in app.state.conversations.conversations.values()
                if c.stage == stage
            ])
            for stage in ConversationStage
        },
    }


@app.post("/admin/trigger-followups", dependencies=[Depends(verify_token)])
async def trigger_followups():
    """Manually trigger follow-up messages."""
    result = daily_follow_ups.delay()
    return {"task_id": result.id, "status": "queued"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
