import numpy as np
import pandas as pd
import Metrica_Viz as mviz
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import glob
from PIL import Image
from IPython.display import Image as IPImage
from IPython.display import display

def PlotPitch(home_pts: np.array,
              away_pts,ball: np.array,
              fig_size: tuple = (12, 7),
              plotHulls: bool=True,
              plotAllPlayers: bool=True,
              title: str='Convex Hulls of Home and Away Teams') -> plt.Figure:
    """
    Function to plot the convex hulls of the home and away teams.
    INPUTS:
    - home_pts: np.array with the coordinates of the home players
    - away_pts: np.array with the coordinates of the away players
    - ball: np.array with the coordinates of the ball
    - fig_size: tuple with the size of the figure
    - plotHulls: boolean to plot the convex hulls
    - plotAllPlayers: boolean to plot all the players or only the vertices
    - title: string with the title of the plot
    OUTPUT:
    - fig: matplotlib figure
    """
    #--------------------------------------------------------------------------------------------------------
    # Call plot_pitch to get fig and ax
    fig, ax = mviz.plot_pitch()
    fig.set_size_inches(fig_size)
    #--------------------------------------------------------------------------------------------------------
    
    # Plotting the convex hulls
    if plotHulls:
        home_hull=ConvexHull(home_pts)
        away_hull=ConvexHull(away_pts)
        for simplex in home_hull.simplices:
            ax.plot(np.array(home_pts)[simplex, 0], np.array(home_pts)[simplex, 1], 'k-')
        for simplex in away_hull.simplices:
            ax.plot(np.array(away_pts)[simplex, 0], np.array(away_pts)[simplex, 1], 'k-')
        #--------------------------------------------------------------------------------------------------------
        # Plotting the vertices and filling the convex hulls
        home_vertices = np.array(home_pts)[home_hull.vertices]
        ax.fill(home_vertices[:, 0], home_vertices[:, 1], 'blue', alpha=0.45, edgecolor='black')
        away_vertices = np.array(away_pts)[away_hull.vertices]
        ax.fill(away_vertices[:, 0], away_vertices[:, 1], 'red', alpha=0.45, edgecolor='black')
    #--------------------------------------------------------------------------------------------------------
    if plotAllPlayers: # (Option 1) Scatter ALL the players
        ax.scatter(home_pts[:,0], home_pts[:,1],
                label='Home Team',
                color='blue', s=100, edgecolor='black', zorder=2)
        ax.scatter(away_pts[:,0], away_pts[:,1],
                label='Away Team',s=100, color='red', edgecolor='black', zorder=2)
    else:  # (Option 2) Scatter ONLY the vertices
        ax.scatter(home_vertices[:, 0], home_vertices[:, 1], label='Home Team',
                color='blue', s=100, edgecolor='black', zorder=2)
        ax.scatter(away_vertices[:, 0], away_vertices[:, 1], label='Away Team',
                s=100, color='red', edgecolor='black', zorder=2)
    #--------------------------------------------------------------------------------------------------------
    # Ball
    ax.scatter(ball[0],ball[1], label='Ball', 
               s=75, color='white', edgecolor='black', zorder=2)
    #--------------------------------------------------------------------------------------------------------
    # Adding legend and title
    ax.legend(loc='lower right', fontsize=12, facecolor='white', edgecolor='black', bbox_to_anchor=(.97, 0.05));
    ax.set_title(title, fontsize=15, fontweight='bold')
    #--------------------------------------------------------------------------------------------------------
    plt.close(fig) # Close the figure to avoid double plotting
    return fig

def PlotGIF(home_xy: pd.DataFrame, 
            away_xy: pd.DataFrame, 
            initial_frame: int=0, 
            final_frame: int=2101) -> None:
    """
    Function to create and display a GIF with the convex hulls of the home and away teams.
    The function saves the images in the figs folder and the GIF in the gifs folder.
    INPUTS:
    - home_xy: pd.DataFrame with the coordinates of the home players
    - away_xy: pd.DataFrame with the coordinates of the away players
    - initial_frame: int with the initial frame to plot
    - final_frame: int with the final frame to plot
    OUTPUT:
    - GIF displayed in the notebook
    """
    # Plotting the convex hulls for all frames from 1000 to 1200
    for frame in range(initial_frame,final_frame):
        #--------------------------------------------------------------------
        # Retrieve the data for the frame
        home_data=home_xy.iloc[frame,:]
        away_data=away_xy.iloc[frame,:]
        home_data=home_data.dropna()
        away_data=away_data.dropna()
        ball=np.array(home_data[-2:])
        home_data= home_data[4:-2] #exclude both the goalkeeper and the ball
        away_data= away_data[4:-2] #exclude both the goalkeeper and the ball
        #--------------------------------------------------------------------
        # divide x and y
        home_data_x=home_data[home_data.index.str.contains('_x')]
        home_data_y=home_data[home_data.index.str.contains('_y')]
        away_data_x=away_data[away_data.index.str.contains('_x')]
        away_data_y=away_data[away_data.index.str.contains('_y')]
        #--------------------------------------------------------------------
        # Coordinates
        home_pts= np.array([[x,y] for x,y in zip(home_data_x,home_data_y)])
        away_pts= np.array([[x,y] for x,y in zip(away_data_x,away_data_y)])
        #--------------------------------------------------------------------
        curr_plot=PlotPitch(home_pts=home_pts, away_pts=away_pts, ball=ball,plotHulls=True,plotAllPlayers=True)
        #save it
        curr_plot.savefig(f'figs/convex_hulls_{frame}.png', dpi=300, bbox_inches='tight')

    # Generate the GIF

    # Load all the saved images
    image_files = sorted(glob.glob('figs/convex_hulls_*.png'), key=lambda x: int(x.split('_')[-1].split('.')[0]))[49:200]

    # Create a list of images
    # Desired GIF dimensions
    gif_width = 1200
    gif_height = 700
    images = [Image.open(image).resize((gif_width, gif_height), Image.Resampling.LANCZOS) for image in image_files]

    # Save as a GIF
    gif_path = 'gifs/convex_hulls.gif'
    images[0].save(gif_path, save_all=True, append_images=images, duration=0.0001, loop=0)

    # Display the GIF
    display(IPImage(gif_path))