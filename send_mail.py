import ezgmail

def send_email(receiver_email='hemanth5498@gmail.com'):
    """
    send_email sends an email to the email address specified in the
    argument.

    Parameters
    ----------
    email_address: email address of the recipient
    subject: subject of the email
    body: body of the email
    """

    
    
    subject= "Dun dun dun dun"
    body="Emotion classified, the detected emotion is also classified information."
    
    ezgmail.send(receiver_email, subject, body, ['images.jpg'] )



if __name__=='__main__':
    print('Starting')
    send_email()
    print('Finished')