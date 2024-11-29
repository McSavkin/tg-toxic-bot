FROM python:3.10-slim
ENV TOKEN='77704366269:AAGPaHWx7ZH0KwAHFREQkaYR7_ypKufHUxE'
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "bot.py"]