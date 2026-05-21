"""
EmailService — async email delivery via Resend REST API.

Uses httpx.AsyncClient directly; no sync blocking.
All public methods are fire-and-forget-safe: callers should catch EmailDeliveryError
and decide whether to propagate or swallow it.
"""
import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_RESEND_URL = "https://api.resend.com/emails"
_TOKEN_EXPIRY_HOURS = 1  # keep in sync with password_reset_token_repository


class EmailDeliveryError(Exception):
    """Raised when the Resend API returns a non-2xx response."""


async def send_password_reset_email(
    to_email: str,
    username: str,
    reset_url: str,
) -> None:
    """
    Send a password reset email to *to_email*.

    Raises EmailDeliveryError on API failure so the caller can decide
    whether to propagate or swallow it.
    """
    if not settings.RESEND_API_KEY:
        logger.warning(
            "EmailService: RESEND_API_KEY not set — email skipped (to=%s)", to_email
        )
        return

    logger.info("EmailService: queuing reset email (to=%s)", to_email)

    payload = {
        "from": settings.MAIL_FROM,
        "to": [to_email],
        "subject": "NEURAVA — Şifrenizi Sıfırlayın",
        "html": _build_reset_html(username, reset_url),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _RESEND_URL,
                json=payload,
                headers={"Authorization": f"Bearer {settings.RESEND_API_KEY}"},
            )
            resp.raise_for_status()
            resend_id = resp.json().get("id", "—")
            logger.info(
                "EmailService: reset email sent (to=%s resend_id=%s)", to_email, resend_id
            )
    except httpx.HTTPStatusError as exc:
        logger.error(
            "EmailService: Resend API error (to=%s status=%s body=%.200s)",
            to_email, exc.response.status_code, exc.response.text,
        )
        raise EmailDeliveryError(
            f"Resend returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        logger.error(
            "EmailService: network error sending reset email (to=%s): %s", to_email, exc
        )
        raise EmailDeliveryError("Network error contacting Resend") from exc


# ── HTML template ─────────────────────────────────────────────────────────────

def _build_reset_html(username: str, reset_url: str) -> str:
    safe_username = username.replace("<", "&lt;").replace(">", "&gt;")
    safe_url = reset_url.replace('"', "%22")

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1.0" />
  <title>NEURAVA — Şifre Sıfırlama</title>
</head>
<body style="margin:0;padding:0;background:#07080f;font-family:'Segoe UI',Helvetica,Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#07080f;min-height:100vh;">
    <tr>
      <td align="center" style="padding:48px 16px;">

        <!-- Card -->
        <table width="100%" cellpadding="0" cellspacing="0"
          style="max-width:520px;background:#0d0e1c;border-radius:20px;
                 border:1px solid rgba(99,102,241,0.18);
                 box-shadow:0 0 60px -10px rgba(99,102,241,0.25);">

          <!-- Top accent line -->
          <tr>
            <td style="height:3px;border-radius:20px 20px 0 0;
                       background:linear-gradient(90deg,#06b6d4,#818cf8,#a78bfa);"></td>
          </tr>

          <!-- Logo + header -->
          <tr>
            <td align="center" style="padding:40px 40px 0;">
              <div style="display:inline-block;margin-bottom:24px;">
                <span style="font-size:22px;font-weight:800;letter-spacing:0.06em;
                             background:linear-gradient(135deg,#67e8f9,#818cf8);
                             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                             background-clip:text;">
                  NEURAVA
                </span>
              </div>
              <h1 style="margin:0 0 8px;font-size:22px;font-weight:700;
                          color:#f1f5f9;letter-spacing:-0.01em;">
                Şifrenizi Sıfırlayın
              </h1>
              <p style="margin:0;font-size:14px;color:#94a3b8;line-height:1.6;">
                Merhaba <strong style="color:#e2e8f0;">{safe_username}</strong>,<br/>
                NEURAVA hesabınız için şifre sıfırlama isteği aldık.
              </p>
            </td>
          </tr>

          <!-- CTA button -->
          <tr>
            <td align="center" style="padding:36px 40px 28px;">
              <a href="{safe_url}"
                 style="display:inline-block;padding:15px 36px;
                        background:linear-gradient(135deg,#0891b2,#6366f1);
                        color:#ffffff;font-size:15px;font-weight:700;
                        text-decoration:none;border-radius:12px;
                        letter-spacing:0.02em;
                        box-shadow:0 8px 32px -4px rgba(99,102,241,0.50);">
                Şifreyi Sıfırla
              </a>
            </td>
          </tr>

          <!-- Divider -->
          <tr>
            <td style="padding:0 40px;">
              <div style="height:1px;background:rgba(99,102,241,0.12);"></div>
            </td>
          </tr>

          <!-- Expiry + safety notice -->
          <tr>
            <td style="padding:28px 40px 36px;">
              <table cellpadding="0" cellspacing="0" width="100%">
                <tr>
                  <td style="padding:0 0 14px;">
                    <div style="display:flex;align-items:flex-start;gap:10px;">
                      <span style="color:#f59e0b;font-size:14px;line-height:1.5;">⏱</span>
                      <p style="margin:0;font-size:13px;color:#94a3b8;line-height:1.6;">
                        Bu bağlantı <strong style="color:#e2e8f0;">{_TOKEN_EXPIRY_HOURS} saat</strong>
                        içinde geçerliliğini yitirecektir.
                      </p>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td>
                    <div style="display:flex;align-items:flex-start;gap:10px;">
                      <span style="color:#64748b;font-size:14px;line-height:1.5;">🔒</span>
                      <p style="margin:0;font-size:13px;color:#64748b;line-height:1.6;">
                        Bu isteği siz göndermediyseniz bu e-postayı görmezden gelebilirsiniz.
                        Hesabınız güvende kalmaya devam edecektir.
                      </p>
                    </div>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Fallback URL -->
          <tr>
            <td style="padding:0 40px 32px;">
              <div style="background:rgba(99,102,241,0.07);border-radius:10px;
                          border:1px solid rgba(99,102,241,0.14);padding:14px 16px;">
                <p style="margin:0 0 4px;font-size:11px;font-weight:600;
                           text-transform:uppercase;letter-spacing:0.08em;color:#475569;">
                  Bağlantı çalışmıyor mu?
                </p>
                <p style="margin:0;font-size:11px;color:#475569;
                           word-break:break-all;line-height:1.5;">
                  {safe_url}
                </p>
              </div>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td align="center" style="padding:0 40px 36px;">
              <p style="margin:0;font-size:12px;color:#334155;line-height:1.6;">
                Bu e-posta NEURAVA tarafından otomatik olarak gönderilmiştir.<br/>
                Lütfen bu adrese yanıt vermeyiniz.
              </p>
            </td>
          </tr>

          <!-- Bottom accent line -->
          <tr>
            <td style="height:3px;border-radius:0 0 20px 20px;
                       background:linear-gradient(90deg,#06b6d4,#818cf8,#a78bfa);"></td>
          </tr>

        </table>
        <!-- /Card -->

      </td>
    </tr>
  </table>
</body>
</html>"""
