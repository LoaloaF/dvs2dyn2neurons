import numpy as np
import psychopy.visual
import psychopy.event
import psychopy.core
import psychopy.monitors


def shifting_grating_sf(mon=psychopy.monitors.Monitor('testMonitor'), screen_n=0, orientation=0, temporal_frequency=2, spatial_frequency=np.arange(0.01, 0.31, 0.01), duration=3):

    """
    To produce a sequence of gratings (full screen) with different spatial frequencies

    :param mon: monitor object (psychopy.monitors.Monitor('monitor_name') (create a monitor in Psychopy monitor center,
    specify number of pixels, width and distance from the screen. This necessary to use "deg" units)
    :param screen_n: index of the screen to be used (0 for current screen, 1 for second screen)
    :param orientation: angle from the vertical direction in degrees
    :param temporal_frequency: temporal frequency in Hz
    :param spatial_frequency: array of values of spatial frequencies in cycles/deg
    :param duration: duration of each stimulus in seconds
    """

    win = psychopy.visual.Window(
        units="deg",
        monitor=mon,
        screen=screen_n,
        fullscr=True,
        size=mon.getSizePix()
    )

    grating = psychopy.visual.GratingStim(
        win=win,

        # Size of the actual grating in pixels, can be adjusted to screen dimension.
        size=[100, 100],

        # Masks can be added: "circle", "gauss", "raisedCos", "cross".
        # To change mask's parameters see maskParams in Psychopy documentation.

        # mask="gauss",

        ori=orientation
    )

    for freq in spatial_frequency:

        grating.sf = freq

        clock = psychopy.core.Clock()

        while clock.getTime() < duration:

            # In Psychopy phase varies between 0 and 1.
            # Assigning a phase value of time*n drifts the stimulus at n Hz.
            grating.phase = np.mod(clock.getTime()*temporal_frequency, 1)

            grating.draw()

            win.flip()

            keys = psychopy.event.getKeys()

            if len(keys) > 0:
                break


def shifting_grating_ori(mon=psychopy.monitors.Monitor('testMonitor'), screen_n=0, orientation=np.arange(-90, 91, 5), temporal_frequency=2, spatial_frequency=0.16, duration=3):

    """
    To produce a sequence of gratings (full screen) with different orientations

    :param mon: monitor object (psychopy.monitors.Monitor('monitor_name') (create a monitor in Psychopy monitor center,
    specify number of pixels, width and distance from the screen. This necessary to use "deg" units)
    :param screen_n: index of the screen to be used (0 for current screen, 1 for second screen)
    :param orientation: array of values of orientations (angle from the vertical direction in degrees)
    :param temporal_frequency: temporal frequency in Hz
    :param spatial_frequency: spatial frequencies in cycles/deg
    :param duration: duration of each stimulus in seconds
    """

    win = psychopy.visual.Window(
        units="deg",
        monitor=mon,
        screen=screen_n,
        fullscr=True,
        size=mon.getSizePix()
    )

    grating = psychopy.visual.GratingStim(
        win=win,

        # Size of the actual grating in pixels, can be adjusted to screen dimension.
        size=[100, 100],

        # Masks can be added: "circle", "gauss", "raisedCos", "cross".
        # To change mask's parameters see maskParams in Psychopy documentation.

        # mask="gauss",

        sf=spatial_frequency
    )

    for angle in orientation:

        grating.ori = angle

        clock = psychopy.core.Clock()

        while clock.getTime() < duration:

            # In Psychopy phase varies between 0 and 1.
            # Assigning a phase value of time*n drifts the stimulus at n Hz.
            grating.phase = np.mod(clock.getTime()*temporal_frequency, 1)

            grating.draw()

            win.flip()

            keys = psychopy.event.getKeys()

            if len(keys) > 0:
                break


def shifting_grating_tf(mon=psychopy.monitors.Monitor('testMonitor'), screen_n=0, orientation=0, temporal_frequency=np.arange(0.3, 4, 0.1), spatial_frequency=0.16, duration=3):

    """
    To produce a sequence of gratings (full screen) with different temporal frequencies

    :param mon: monitor object (psychopy.monitors.Monitor('monitor_name') (create a monitor in Psychopy monitor center,
    specify number of pixels, width and distance from the screen. This necessary to use "deg" units)
    :param screen_n: index of the screen to be used (0 for current screen, 1 for second screen)
    :param orientation: angle from the vertical direction in degrees
    :param temporal_frequency: array of values of temporal frequencies in Hz
    :param spatial_frequency: spatial frequencies in cycles/deg
    :param duration: duration of each stimulus in seconds
    """

    win = psychopy.visual.Window(
        units="deg",
        monitor=mon,
        screen=screen_n,
        fullscr=True,
        size=mon.getSizePix()
    )

    grating = psychopy.visual.GratingStim(
        win=win,

        # Size of the actual grating in pixels, can be adjusted to screen dimension.
        size=[100, 100],

        # Masks can be added: "circle", "gauss", "raisedCos", "cross".
        # To change mask's parameters see maskParams in Psychopy documentation.

        # mask="gauss",

        sf=spatial_frequency,
        ori=orientation
    )

    for tf in temporal_frequency:

        clock = psychopy.core.Clock()

        while clock.getTime() < duration:

            # In Psychopy phase varies between 0 and 1.
            # Assigning a phase value of time*n drifts the stimulus at n Hz.
            grating.phase = np.mod(clock.getTime()*tf, 1)

            grating.draw()

            win.flip()

            keys = psychopy.event.getKeys()

            if len(keys) > 0:
                break


def shifting_bar_v(mon=psychopy.monitors.Monitor('testMonitor'), screen_n=0, orientation=0, vel=np.arange(4, 9, 2), color=[-1, -1, -1], dimension=[2, 100], duration=5):

    """
    To produce a sequence of shifting bars (full screen with grey background) with different velocities

    :param mon: monitor object (psychopy.monitors.Monitor('monitor_name') (create a monitor in Psychopy monitor center,
    specify number of pixels, width and distance from the screen. This necessary to use "deg" units)
    :param screen_n: index of the screen to be used (0 for current screen, 1 for second screen)
    :param orientation: angle from the vertical direction in degrees
    :param vel: array of values of velocities measured in deg/s
    :param color: color of the bar specified by triplet of values ([-1, -1, -1] for black and [1, 1, 1] for white)
    :param dimension: dimension of the bar specified by a pair of values [width, height], measured in deg of visual field
    :param duration: duration of each stimulus measured in sec
    NB: the function has been implemented so that the bar starts moving from the center of the window and doesn't
    reappear once it has exited the window
    """


    
    start = (2, -12)

    # On screen 2
    win1 = psychopy.visual.Window(
        units="deg",
        monitor=mon,
        screen=2,
        fullscr=True,
        size=mon.getSizePix()
    )
    win1.setColor([1,1,1], colorSpace='rgb')
    
    # On screen 3
    win2 = psychopy.visual.Window(
        units="deg",
        monitor=mon,
        screen=3,
        fullscr=True,
        size=mon.getSizePix()
    )
    win2.setColor([1,1,1], colorSpace='rgb')
    
    bar1 = psychopy.visual.Rect(
        win=win1,
        width=dimension[0],
        height=dimension[1],
        ori=orientation,
        fillColor=color,
        lineColor=color
    )

    bar2 = psychopy.visual.Rect(
        win=win2,
        width=dimension[0],
        height=dimension[1],
        ori=orientation,
        fillColor=color,
        lineColor=color
    )
    
    for velocity in vel:

        v_x = velocity * np.cos(np.radians(bar1.ori))
        v_y = -velocity * np.sin(np.radians(bar1.ori))
        end   =  (v_x * duration +start[0],
                   v_y * duration +start[1])
            
        clock = psychopy.core.Clock()        
        while clock.getTime() < duration:

            bar1.pos = [v_x * clock.getTime()+start[0],
                       v_y * clock.getTime()+start[1]]

            bar1.draw()
            
            win1.flip()

            bar2.pos = [v_x * clock.getTime()+start[0],
                       v_y * clock.getTime()+start[1]]

            bar2.draw()

            win2.flip()
            
            keys = psychopy.event.getKeys()
            if len(keys) > 0:
                break

        clock = psychopy.core.Clock()        
        while clock.getTime() < duration:
            
            bar1.pos = [-v_x * clock.getTime()+end[0],
                       v_y * clock.getTime()+end[1]]

            bar1.draw()
            
            win1.flip()

            bar2.pos = [-v_x * clock.getTime()+end[0],
                       v_y * clock.getTime()+end[1]]

            bar2.draw()

            win2.flip()
            
            keys = psychopy.event.getKeys()
            
            if len(keys) > 0:
                break   
            
def shifting_bar_RAINR(mon=psychopy.monitors.Monitor('testMonitor'), screen_n=0, orientation=0, vel=[6]*2, duration=2, color=[-1, -1, -1], dimension=[2, 100]):
    """ Generate stimuli used for RAINR paper dataset """
    
    start = (2, -12)

    # 1 screen only
    win = psychopy.visual.Window(
        units="deg",
        monitor=mon,
        screen=1,
        fullscr=True,
        size=mon.getSizePix()
    )
    win.setColor([1,1,1], colorSpace='rgb')
        
    bar = psychopy.visual.Rect(
        win=win,
        width=dimension[0],
        height=dimension[1],
        ori=orientation,
        fillColor=color,
        lineColor=color
    )
    
    for velocity in vel:

        v_x = velocity * np.cos(np.radians(bar.ori))
        v_y = -velocity * np.sin(np.radians(bar.ori))
        end   =  (v_x * duration +start[0],
                   v_y * duration +start[1])
            
        clock = psychopy.core.Clock()        
        while clock.getTime() < duration:

            bar.pos = [v_x * clock.getTime()+start[0],
                       v_y * clock.getTime()+start[1]]

            bar.draw()
            
            win.flip()

            keys = psychopy.event.getKeys()
            if len(keys) > 0:
                break  

def main():
    #shifting_bar_v(vel=[6]*400, duration=2)
    shifting_bar_RAINR()