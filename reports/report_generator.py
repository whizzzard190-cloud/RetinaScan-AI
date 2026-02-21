from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import datetime


def generate_pdf(image_path, heatmap_path, result, output_path="reports/report.pdf"):

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "RetinaScan-AI Medical Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, f"Generated: {datetime.datetime.now()}")

    c.drawImage(image_path, 50, height - 350, width=200, height=200)
    c.drawImage(heatmap_path, 300, height - 350, width=200, height=200)

    y = height - 380

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Diagnosis:")
    c.setFont("Helvetica", 12)
    c.drawString(150, y, result["class"])

    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Confidence:")
    c.setFont("Helvetica", 12)
    c.drawString(150, y, f'{result["confidence"]} %')

    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Recommendation:")

    c.setFont("Helvetica", 11)
    c.drawString(50, y - 20, "Please consult an ophthalmologist for clinical confirmation.")

    c.showPage()
    c.save()

    return output_path