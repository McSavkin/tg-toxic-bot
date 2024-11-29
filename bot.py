import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters.command import Command

# import pandas as pd

import torch

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# df = pd.read_csv('data/labeled.csv')


model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

    
def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba


# Initializing
# TOKEN = os.getenv('TOKEN')
bot = Bot(token='7704366269:AAGPaHWx7ZH0KwAHFREQkaYR7_ypKufHUxE')
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

# Start processing

@dp.message(Command(commands=['start']))
async def proccess_command_start(massage: Message):
    user_name = massage.from_user.full_name
    user_id = massage.from_user.id
    text = f'Привет, {user_name} введи предложение для проверки на токсичность!'
    logging.info(f'{user_name} {user_id} запустил бота')
    await bot.send_message(chat_id=user_id, text=text)


# Predictions

# @dp.message(Command(commands=['get prediction']))
# async def proccess_command_start(massage: Message):
#     user_name = massage.from_user.full_name
#     user_id = massage.from_user.id
#     text = f'{user_name} введи предложение для проверки на токсичность'
#     await bot.send_message(chat_id=user_id, text=text)
#     text = f'Привет, {user_name}!'
#     logging.info(f'{user_name} {user_id} запустил бота')
#     await bot.send_message(chat_id=user_id, text=text)


# All messages

@dp.message()
async def send_prediction(massage: Message):
    user_name = massage.from_user.full_name
    user_id = massage.from_user.id
    text = massage.text
    logging.info(f'{user_name} {user_id}: {text}')
    await massage.answer(text='Текст токсичный с вероятностью - ' + str(text2toxicity(text, True)))

# Polling

if __name__ == '__main__':
    dp.run_polling(bot)
