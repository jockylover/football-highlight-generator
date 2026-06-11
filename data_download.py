from SoccerNet.Downloader import SoccerNetDownloader

from config import SOCCERNET_DIR


def main():
    downloader = SoccerNetDownloader(LocalDirectory=str(SOCCERNET_DIR))
    downloader.password = "s0cc3rn3t"  # SoccerNet 公开数据访问口令
    # downloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["valid"])
    # downloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])
    # downloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])
    downloader.downloadGames(files=["Labels-v2.json"], split=["train"])


if __name__ == "__main__":
    main()
