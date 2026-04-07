import os
from email.message import EmailMessage
import ssl
import smtplib
from datetime import datetime, timedelta
import pytz
import pyotp
import time


   
def send_email(email_receiver, subject, body):
    # email sender, that will send email to users
    email_sender = "test@gmail.com"
    email_password = ''
    email_receiver = email_receiver

    subject = subject
    body = body
    em = EmailMessage()
    em["FROM"] = email_sender
    em["TO"] = email_receiver
    em["SUBJECT"] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    # using SMTP to login as the email sender
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())
        