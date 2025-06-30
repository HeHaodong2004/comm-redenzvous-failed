import matplotlib.pyplot as plt
from copy import deepcopy

from env import Env
from agent import Agent
from utils import *
from node_manager import NodeManager

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class MultiAgentWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.n_agent = N_AGENTS
        self.node_manager = NodeManager(plot=self.save_image)

        self.robot_list = [Agent(i, policy_net, self.node_manager, self.device, self.save_image) for i in
                           range(N_AGENTS)]

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(15):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        # —— 初始化：用各自的 agent_belief 构建图 —— #
        for i, robot in enumerate(self.robot_list):
            map_info_i = self.env.get_agent_map(i)
            robot.update_graph(map_info_i, self.env.robot_locations[i])
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)

        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            for robot in self.robot_list:
                observation = robot.get_observation()
                robot.save_observation(observation)

                next_location, next_node_index, action_index = robot.select_next_waypoint(observation)
                robot.save_action(action_index)

                node = robot.node_manager.nodes_dict.find((robot.location[0], robot.location[1]))
                check = np.array(node.data.neighbor_list)
                assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location,
                                                                                                         robot.location,
                                                                                                         node.data.neighbor_list)
                assert next_location[0] != robot.location[0] or next_location[1] != robot.location[1]

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            # solve collision
            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[id].node_manager.nodes_dict.nearest_neighbors(
                        selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            continue
                        selected_location = coords
                        break

                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location

            reward_list = []
            # —— 1) 对所有 agent 应用 env.step（完成 Belief 更新 + 组内合并） —— #
            for robot, loc in zip(self.robot_list, selected_locations):
                self.env.step(loc, robot.id)

            # —— 2) 用各自的、合并后的 agent_belief 重建图并计算 reward —— #
            reward_list = []
            for idx, (robot, next_node_index) in enumerate(zip(self.robot_list, next_node_index_list)):
                # 拿到“通信受限后”第 idx 个 agent 的 MapInfo
                map_info_i = self.env.get_agent_map(idx)
                robot.update_graph(map_info_i, self.env.robot_locations[idx])

                # 计算个体奖励
                individual_reward = robot.utility[next_node_index] / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
                reward_list.append(individual_reward)

            if self.robot_list[0].utility.sum() == 0:
                done = True

            team_reward = self.env.calculate_reward() - 0.5
            if done:
                team_reward += 10

            for robot, reward in zip(self.robot_list, reward_list):
                robot.save_reward(reward + team_reward)
                robot.update_planning_state(self.env.robot_locations)
                robot.save_done(done)

            if self.save_image:
                self.plot_env(i)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = self.env.get_total_travel()
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save episode buffer
        for robot in self.robot_list:
            observation = robot.get_observation()
            robot.save_next_observations(observation)
            for i in range(len(self.episode_buffer)):
                self.episode_buffer[i] += robot.episode_buffer[i]

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def plot_env(self, step):
        plt.switch_backend('agg')
        # overall belief and trajectories
        color_list = ['r', 'b', 'g', 'y']
        fig, axes = plt.subplots(1, 2 + self.n_agent, figsize=(5 * (2 + self.n_agent), 5))

        # Panel 1: global belief + robot poses
        ax_global = axes[0]
        #ax_global.imshow(self.env.robot_belief, cmap='gray')
        ax_global.imshow(self.env.global_belief, cmap='gray')
        ax_global.axis('off')
        for robot in self.robot_list:
            c = color_list[robot.id]
            cell = get_cell_position_from_coords(robot.location, robot.map_info)
            ax_global.plot(cell[0], cell[1], c+'o', markersize=12, zorder=5)
            ax_global.plot((np.array(robot.trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                           (np.array(robot.trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size,
                           c, linewidth=2, zorder=1)
        ax_global.set_title('Global Belief')

        # Panel 2: global frontiers & graph for agent 0
        #ax0 = axes[1]
        #ax0.imshow(self.env.robot_belief, cmap='gray')
        ax0 = axes[1]
        agent0 = self.robot_list[0]
        ax0.imshow(agent0.map_info.map, cmap='gray')   # 用 agent0 的 map_info
        ax0.axis('off')
        for coords in self.robot_list[0].node_coords:
            node = self.node_manager.nodes_dict.find(coords.tolist()).data
            for neigh in node.neighbor_list[1:]:
                mid = (np.array(neigh) + coords) / 2
                ax0.plot(( [coords[0], mid[0]] - self.robot_list[0].map_info.map_origin_x) / self.robot_list[0].map_info.cell_size,
                         ( [coords[1], mid[1]] - self.robot_list[0].map_info.map_origin_y) / self.robot_list[0].map_info.cell_size,
                         'tan', zorder=1)
        ax0.set_title('Agent 0 Graph')

        # Panels 3...: per-agent local observation maps
        for i, robot in enumerate(self.robot_list):
            ax = axes[2 + i]
            # use the agent's updating_map_info (local view)
            obs_map = robot.updating_map_info.map
            ax.imshow(obs_map, cmap='gray')
            ax.set_title(f'Agent {robot.id} Observation')
            ax.axis('off')

        # suptitle and save
        fig.suptitle(f'Step {step} | Explored: {self.env.explored_rate:.3g} | Dist: {self.env.get_total_travel()}')
        plt.tight_layout()
        out_path = f'{gifs_path}/{self.global_step}_{step}_obs.png'
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        self.env.frame_files.append(out_path)
