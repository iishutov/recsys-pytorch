from pandas import DataFrame

def down_imgs(movie_ids: list[int]):
    import requests
    
    for movie_id in movie_ids:
        img_url = f'https://recsysart.ru/posters/{movie_id}.jpg'
        img = requests.get(img_url).content
        with open(f'./data/posters/{movie_id}.jpg', 'wb') as f:
            f.write(img)

def recs_html(rec_ids: list[int], df_items: DataFrame,
              height: int = 100, title: str = '') -> str:
    from os.path import isfile
    
    download_ids = [rec_id for rec_id in rec_ids if not isfile(f'./data/posters/{rec_id}.jpg')]
    down_imgs(download_ids)

    imgs_html = ''
    for item_id in rec_ids:
        imgs_html += f"""
        <tr>
        <td><img src="data/posters/{item_id}.jpg" style="height: {height}; min-width: 100;"></td>
        <td>""" + \
        df_items[df_items['item_id'] == item_id]['title'].values[0] + \
        """</td>
        <td>
        """ + \
        df_items[df_items['item_id'] == item_id]['description'].values[0] + \
        """</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <title>Recs</title>
        </head>
        <body>
            <h1 style="text-align: center;">{title}</h1>
            <table>
            {imgs_html}
            </table>
        </body>
    </html>
    """

    return html