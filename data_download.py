import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory=r"E:/System Default/table/paper/data/SoccerNet/train")
mySoccerNetDownloader.password = "s0cc3rn3t"
# mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["valid"])
# mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])
# mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train"])
