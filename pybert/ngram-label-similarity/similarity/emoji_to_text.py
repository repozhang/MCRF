import demoji
# demoji.download_codes()
def del_emoji(data):
    #
    replace=demoji.replace(data)
    return(replace)

if __name__=="__main__":
    tweet = "startspreadingthenews yankees win great start by 🎅🏾 going 5strong innings with 5k’s🔥 🐂"
    print(del_emoji(tweet))
    """
    return: startspreadingthenews yankees win great start by  going 5strong innings with 5k’s 
    """
