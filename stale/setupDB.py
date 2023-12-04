import json
from pymongo import MongoClient
from datasets import load_dataset


myclient = MongoClient("localhost")
mydb = myclient["kilt"]
mycol = mydb["knowledgesource"]
breakpoint()
