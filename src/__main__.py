import database as db
import pandas as pd

account_dataset, client_dataset, disp_dataset, district_dataset = db.parse_data()

print(db.select_account(576))
