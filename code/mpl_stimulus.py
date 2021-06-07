from matplotlib import pyplot as plt
from matplotlib import animation

def LED(hz, size=1):
    frames = []
    fig, ax = plt.subplots(figsize=(19,11), facecolor='gray')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    ax.set_facecolor('gray')

    art = ax.plot(1, .5, 'o', markersize=200*size, color='white')
    frames.append(art)

    art = ax.plot(1, .5, 'o', markersize=200*size, color='black')
    frames.append(art)
    
    ani = animation.ArtistAnimation(fig, frames, interval=1000//hz, blit=True)
    plt.show()

def main():
    hz = 60
    size = 1
    LED(hz=hz, size=size)

if __name__ == '__main__':
    main()