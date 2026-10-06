import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

outer = MIMEMultipart("related")
outer["From"] = "shop@news.example.com"
outer["Subject"] = "Your account is locked"
outer.attach(MIMEText("Verify your password immediately.", "plain"))
outer.attach(MIMEText(
    '<html><body><p>Your account is locked. Verify password immediately.</p>'
    '<img src="cid:logo123">'
    '<a href="https://example.com/login">Click here</a></body></html>', "html"))
img = MIMEImage(b"\x89PNG fake image bytes", _subtype="png")
img.add_header("Content-ID", "<logo123>")
outer.attach(img)

print(base64.b64encode(outer.as_bytes()).decode())