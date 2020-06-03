#Submitting to kaggle competitions
#It is assumed that the json file is already uploaded
#in folder named as .kaggle

# connect to kaggle api
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi({"username":"kprokopi","key":"9a28305a56476ed46a0ac149764d9610"})
api.authenticate()

# Signature: competition_submit(file_name, message, competition,quiet=False)
api.competition_submit("sub.csv", "my submission message", "Instacart-Market-Basket-Analysis")

#leaderboard = api.competition_view_leaderboard("Instacart-Market-Basket-Analysis")
#print(leaderboard)
