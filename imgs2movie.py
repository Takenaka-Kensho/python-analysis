import imageio.v2 as imageio
from tqdm.auto import tqdm

"""##################################################################################
    imgs2movie.py

    ***EXPLANATION***
    translate images of contour to mp4

    ***VERSION***
    v2.0 (2025/07)
    - optimize the code to run on the PS
    - replace cv2 to imageio (because cv2 is unable to use in PS)

    ***STRUCTURE***
    .
    ├──GKNET.data  (IN)
    └──arbitary directory/
        ├──imgs2movie.py
        └──mp4 file  (OUT)
##################################################################################"""

def img2mv():
    ##############################################
    pmode = 0                           # 0: phi, 1: A_parallel
    stnum = 0
    ednum = 128
    pngdc = 'phi'
    mode  = 3
    mfps  = 10.0                        # frames per sec
    sname = ''                          # save name is {sname}{mode}P.mp4, so you can also assign empty sentence ''
    ##############################################
    print(f'start num: {stnum:>5}')
    print(f'end num  : {ednum:>5}')
    print(f'mode(wn) : {mode:>5}')
    print(f'fps      : {mfps:>5}')
    print('\033[33m' + 'reading images......' + '\033[0m')
    if pmode == 0:   img = imageio.imread(f'./{pngdc}/{str(mode).zfill(2)}P{str(stnum).zfill(4)}.png')    # 先頭の画像を読み込み
    elif pmode == 1: img = imageio.imread(f'./{pngdc}/{str(mode).zfill(2)}A{str(stnum).zfill(4)}.png')    # 先頭の画像を読み込み
    height, width, layers = img.shape
    size = (width, height)
    if pmode == 0:   savename = f'{sname}{str(mode).zfill(2)}P.mp4'
    elif pmode == 1: savename = f'{sname}{str(mode).zfill(2)}A.mp4'

    imgs = []
    for j in tqdm(range(stnum, ednum)):
        if pmode == 0:   filename = f'./{pngdc}/{str(mode).zfill(2)}P{str(j).zfill(4)}.png'
        elif pmode == 1: filename = f'./{pngdc}/{str(mode).zfill(2)}A{str(j).zfill(4)}.png'
        img = imageio.imread(filename)
        if img is None:     # 読み込みエラー処理
            print(filename + 'is not found')
            break
        else:
            imgs.append(img)
    print('\033[33m' + 'translating to mp4......' + '\033[0m')
    imageio.mimsave(savename, imgs, fps=mfps)
    # out.release()
    print('\033[33m'+f'saved as "{savename}"'+'\033[0m')

if __name__ == "__main__":
    print('\033[33m'+'='*80+'\033[0m')
    img2mv()
    print('\033[33m'+'='*80+'\033[0m')