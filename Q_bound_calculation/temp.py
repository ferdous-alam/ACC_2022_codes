from visualization.visualizations_FR import Visualizations

from calculate_bound import *


all_L_terms = []
all_R_terms = []
Delta_list = 3  # [1, 2, 3, 4, 5, 6, 7, 8]
gamma_list = [0.99, 0.97, 0.95, 0.93, 0.91, 0.89, 0.87, 0.85]

for i in range(len(gamma_list)):
    print(f'exp#{i+1}')
    Delta = Delta_list
    gamma = gamma_list[i]
    # source
    env_type = ['source', 'original', 'original']
    V_o_source, rewards_source = get_option_value(env_type, gamma, Delta)
    print(' ########### finished! ################ ')

    # target
    env_type = ['target', 'v1', 'v1']
    V_o_target, rewards_target = get_option_value(env_type, gamma, Delta)
    print(' ########### finished! ################ ')
    V_o_diff = np.subtract(V_o_target, V_o_source)
    r_o_diff = np.subtract(rewards_target, rewards_source)
    V_supremum = abs(np.max(V_o_diff))
    r_supremum = abs(np.max(r_o_diff))
    R_term = r_supremum / (1 - (gamma ** Delta))
    L_term = V_supremum

    all_L_terms.append(L_term)
    all_R_terms.append(R_term)

all_L_terms = all_L_terms/max(all_R_terms)
all_R_terms = all_R_terms/max(all_R_terms)

viz = Visualizations()
viz.plot_bounds(all_L_terms, all_R_terms, gamma_list)

