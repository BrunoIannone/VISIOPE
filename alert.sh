#!/bin/bash
MSMTP_CONFIG_FILE="/home/USER_NAME/.msmtprc"




recipients="YOUR_EMAIL_HERE"
subject="Training complete!"

echo -e "Subject:$subject\nFrom:email_here\n\nHi! training was completed. Come and join me!" | sudo -u USER_NAME msmtp --file=$MSMTP_CONFIG_FILE -a gmail $recipients

exit 0

