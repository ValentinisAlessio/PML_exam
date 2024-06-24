import numpy as np
import Metrica_Viz as mviz
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def PlotPitch(home_pts, away_pts,ball,
              fig_size=(12, 7),
              plotHulls=True,plotAllPlayers=True,title='Convex Hulls of Home and Away Teams'):
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