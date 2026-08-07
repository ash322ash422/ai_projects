# import os
import smtplib
# from pathlib import Path
from email.message import EmailMessage
from azure.communication.email import EmailClient


import sys
from pathlib import Path

# Add app directory to path FIRST
sys.path.append(str(Path(__file__).resolve().parent.parent))


from config import (GMAIL_EMAIL, 
                    GMAIL_APP_PASSWORD, 
                    AZURE_EMAIL_COMMUNICATIONS_STRING, 
                    AZURE_EMAIL_SENDER
)

def send_email_through_gmail(
    to_email: str,
    subject: str,
    body: str
):
    """
    Send an email using Gmail SMTP.
    """

    msg = EmailMessage()

    msg["From"] = GMAIL_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:

        smtp.starttls()

        smtp.login(
            GMAIL_EMAIL,
            GMAIL_APP_PASSWORD
        )

        smtp.send_message(msg)

    print("Email sent successfully.")


def build_success_email(download_url: str) -> tuple[str, str]:
    """
    Returns the subject and body for a successful tender processing email.
    """

    subject = "Tender Processing Completed"

    body = f"""
            Hello,

            Your tender has been processed successfully.

            You can download the generated Excel file using the link below:

            {download_url}

            It is valid for the next 24 hours.
            Thank you for using the Tender Automation System.

            Regards,
            Tender Automation System
    """

    return subject, body

def send_email_through_azure( recipient, subject, body):

    client = EmailClient.from_connection_string(AZURE_EMAIL_COMMUNICATIONS_STRING)

    message = {
        "senderAddress": AZURE_EMAIL_SENDER,
        "recipients": {
            "to": [
                {
                    "address": recipient
                }
            ]
        },

        "content": {
            "subject": subject,
            "plainText": body
        }

    }

    poller = client.begin_send(message)

    result = poller.result()

    print(result)




if __name__ == "__main__":

    download_url = (
        "https://storetender.blob.core.windows.net/automation-file-processed/01_tender_mini_version_CLEAN.xlsx?se=2026-08-07T15%3A42%3A33Z&sp=r&sv=2026-06-06&sr=b&sig=Z0qXAsOvetg8qaIYp92hdHw3qQkgWTlM32jzFtMZx%2BQ%3D"
    )
    subject, body = build_success_email(download_url)
    
    
    # send_email_through_gmail(
    #     to_email="ash322.ash422@gmail.com",
    #     subject=subject,
    #     body=body
    # )
    
    send_email_through_azure(
        recipient="ash322.ash422@gmail.com",
        subject=subject,
        body=body
    )
    