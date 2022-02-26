from visualization.visualizations_FR import Visualizations


viz = Visualizations()
all_L_terms = [5, 6, 7, 8, 9, 10]
all_R_terms = [1, 2, 3, 4, 5, 6]
Delta_list = [1, 2, 3, 4, 5, 6]

viz.plot_bounds(all_L_terms, all_R_terms, Delta_list)
