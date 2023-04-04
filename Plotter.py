import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import plotly.graph_objects as go
import discreteaction_pendulum
from agent_network import DQN
import plotly.io as pio
from plotly.subplots import make_subplots
from hyperparameters import *

env = discreteaction_pendulum.Pendulum()
n_observations = env.num_states
n_actions = env.num_actions
mode=3
weight_dir=weight_dir+'/'+str(mode)+'/'
reward_file = 'rewards.txt'
fig_dir=fig_dir+'/'+str(mode)+'/'
weight_file   = weight_dir + 'q_m_w_e.pth'
plotter       = Plotter.Plotter(env, weight_file, hidden_size, device)
plotter.plot_torque_vs_vel(fig_dir + 'torque_vs_vel.png')

plotter.plot_learning_curve(gamma, reward_dir + reward_file, 
                            fig_dir + 'learning_curve.png')
plotter.plot_policy(fig_dir + 'policy.png')
plotter.plot_value_fn(fig_dir + 'value_fn.png')
plotter.plot_trajectory(fig_dir + 'trajectory.png', fig_dir + 'trajectory_vfn.png')

legend_list   = ['returns', 
                  'returns (no replay)', 
                  'returns (no target)',
                  'returns (no replay, no target)']
file_list     = [ reward_dir + 'rewards.txt', 
                  reward_dir + 'rewards_no_rep.txt', 
                  reward_dir + 'rewards_no_tar.txt',
                  reward_dir + 'rewards_no_tar_no_rep.txt']


plotter.plot_ablation_study(gamma, file_list, legend_list, 
                            test_dir + 'figures/ablation_study1.png')


weight_list   = [ test_dir + 'NN_weights/target_replay/qnet_model_weights_end.pth',
                  test_dir + 'NN_weights/no_replay/qnet_model_weights_end.pth', 
                  test_dir + 'NN_weights/no_target/qnet_model_weights_end.pth',
                  test_dir + 'NN_weights/no_target_no_replay/qnet_model_weights_end.pth']

plotter.plot_ablation_study_bar(500, gamma, weight_list, test_dir + 
                                      'figures/ablation_study_bar2.png')

class Plotter:
    def __init__(self, env, weight_file, hidden_size, device):
        self.env = env
        self.NN  = DQN(env.num_states, env.num_actions, hidden_size).to(device)
        self.NN.load_state_dict(torch.load(weight_file))
        self.device = device
        self.simulate()
    
    def simulate(self):
        self.n        = 100
        self.theta    = torch.linspace(-np.pi, np.pi, self.n)
        self.thetadot = torch.linspace(-self.env.max_thetadot, self.env.max_thetadot, 
                                  self.n)
        state         = torch.cat((self.theta, self.thetadot)).reshape(self.n, 2)\
                        .to(self.device)
        
        self.tau      = np.zeros((self.n, self.n))
        self.q_vals   = np.zeros((self.n, self.n))
        
        with torch.no_grad():
            for i in range(self.n):
                for j in range(self.n):
                    state             = torch.tensor([self.theta[j], 
                                                      self.thetadot[i]])
                    out               = self.NN(state).detach().numpy()
                    self.tau[i, j]    = self.env._a_to_u(np.argmax(out))
                    self.q_vals[i, j] = np.max(out)
                
    def policy(self, state):
        state  = torch.tensor(state, dtype=torch.float32, device= self.device)\
                 .unsqueeze(0)
        with torch.no_grad():
            action = self.NN(state).max(1)[1].view(1, 1).item()
        return action
    
    def plot_torque_vs_vel(self, dest_file):


        # Set dark grid style
        layout = go.Layout(
            template='plotly_dark',
            xaxis_title=r"Angular velocity, $\dot \theta$",
            yaxis_title=r"Torque, $\tau$",
            legend=dict(
                x=0.7,
                y=1.1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="white"
                ),
                bgcolor="#1f2c56",
                bordercolor="black",
                borderwidth=2
            ),
            margin=dict(l=60, r=30, t=60, b=30),
            width=500,
            height=500
        )

        idx = int(self.n/2)
        torque = self.tau[:, idx]
        theta = round(float(self.theta[idx]), 2)

        fig = go.Figure(layout=layout)
        fig.add_trace(go.Scatter(x=self.thetadot.numpy(), y=torque,
                                mode='lines',
                                name=r'$\theta$ = {}'.format(theta),
                                line=dict(width=2, color='red')
                                ))
        fig.update_layout(
            legend_title='<b>Theta</b>'
        )
        fig.write_image(dest_file)
        fig.show()
        
        
    def plot_learning_curve(self, gamma, source_file, dest_file):
        with open(source_file) as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]

        # Split each line into a list of integers
        episode_list = [[int(num) for num in line[1:-1].split(',')] 
                        for line in lines]
        num_eps              = len(episode_list)
        discounted_returns   = np.zeros(num_eps)
        undiscounted_returns = np.zeros(num_eps)
        for e in range(num_eps):
            returns   = 0
            u_returns = 0
            for i in range(len(episode_list[e])):
                returns   += (gamma**i) * episode_list[e][i]
                u_returns += episode_list[e][i]
            discounted_returns[e]   = returns
            undiscounted_returns[e] = u_returns
        
        mean_ud_returns = []
        sd_ud_returns   = []
        
        mean_d_returns  = []
        sd_d_returns    = []
        for i in range(num_eps):
            if i < 100:
                mean_ud_returns.append(np.mean(undiscounted_returns[0:i+1]))
                sd_ud_returns.append(np.std(undiscounted_returns[0:i+1]))
                
                mean_d_returns.append(np.mean(discounted_returns[0:i+1]))
                sd_d_returns.append(np.std(discounted_returns[0:i+1]))
                
            else:
                mean_ud_returns.append(np.mean(undiscounted_returns[i-100:i]))
                sd_ud_returns.append(np.std(undiscounted_returns[i-100:i]))
                
                mean_d_returns.append(np.mean(discounted_returns[i-100:i]))
                sd_d_returns.append(np.std(discounted_returns[i-100:i]))
                
        
        mean_d_returns = np.array(mean_d_returns)
        sd_d_returns = np.array(sd_d_returns)
        mean_ud_returns = np.array(mean_ud_returns)
        sd_ud_returns = np.array(sd_ud_returns)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        colors = ['rgba(0, 100, 80, 0.2)', 'rgba(0, 100, 80, 0.4)', 'rgba(0, 100, 80, 0.6)', 'rgba(0, 100, 80, 0.8)', 'rgba(0, 100, 80, 1)']

        fig.add_trace(go.Scatter(x=list(range(num_eps)), y=mean_d_returns, mode='lines', name='Discounted Returns Mean', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(num_eps)), y=mean_ud_returns, mode='lines', name='Undiscounted Returns Mean', line=dict(color='blue')), row=2, col=1)

        lower = mean_d_returns - sd_d_returns
        upper = mean_d_returns + sd_d_returns

        fig.add_trace(go.Scatter(x=list(range(num_eps)) + list(range(num_eps)[::-1]), y=list(lower) + list(upper)[::-1], fill='toself', fillcolor=colors[0], line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False), row=1, col=1)

        lower = mean_ud_returns - sd_ud_returns
        upper = mean_ud_returns + sd_ud_returns

        fig.add_trace(go.Scatter(x=list(range(num_eps)) + list(range(num_eps)[::-1]), y=list(lower) + list(upper)[::-1], fill='toself', fillcolor=colors[0], line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False), row=2, col=1)
        fig.update_layout(xaxis=dict(title='Episodes'))
        fig.update_yaxes(title_text='Mean Discounted Returns', row=1, col=1, range=[-15, 20])
        fig.update_yaxes(title_text='Mean Undiscounted Returns', row=2, col=1, range=[-110, 110])
        fig.update_layout(title_text='DQN Learning Curve', title_x=0.5, height=800)

        fig.update_layout(
            template='plotly_dark',  
            font=dict(
                family='Arial',  
                size=18  
            ),
            paper_bgcolor='rgb(0,0,0)', 
            plot_bgcolor='rgb(0,0,0)'  
        )

        fig.write_image(dest_file)
        fig.show()
            
            
        
    
    def plot_policy(self, filename):
        x, y = np.meshgrid(self.theta, self.thetadot)
        z = self.tau
        z_min, z_max = np.min(z), np.max(z)

        fig = go.Figure(data=go.Heatmap(
            x=self.theta,
            y=self.thetadot.numpy(),
            z=z,
            zmin=z_min,
            zmax=z_max,
            colorscale='Viridis',
            colorbar=dict(
                title='Tau',
                titleside='right',
                titlefont=dict(size=14),
                tickfont=dict(size=12),
                lenmode='fraction',
                len=0.4
            ),
        ))
        fig.update_layout(
            title=dict(
                text='Heatmap of Tau vs Theta and Theta_dot',
                font=dict(size=20)
            ),
            xaxis=dict(
                title=r"$\theta$",
                titlefont=dict(size=16),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=r"$\dot \theta$",
                titlefont=dict(size=16),
                tickfont=dict(size=14)
            ),
            margin=dict(l=80, r=40, t=80, b=40),
            width=900,
            height=700,
            font=dict(
                family="Arial",
                size=16,
                color="white"
            ),
            template='plotly_dark'
        )
        fig.show()
        fig.write_image(filename)
        # pio.write_image(fig, filename, format='png', width=800, height=600, scale=2)



    def plot_value_fn(self, filename):
        x, y = np.meshgrid(self.theta, self.thetadot)
        z = self.q_vals

        fig = go.Figure(data=go.Heatmap(x=x.flatten(), y=y.flatten(), z=z.flatten(),
                                        colorscale='Viridis',
                                        zmin=np.min(z),
                                        zmax=np.max(z),
                                        colorbar=dict(title='Value function', tickfont=dict(size=12))))

        fig.update_layout(title='Value function', xaxis_title=r'$\theta$', yaxis_title=r'$\dot{\theta}$',
                        font=dict(size=16), width=800, height=600, template='plotly_dark')
        fig.update_coloraxes(colorbar=dict(tickfont=dict(size=12)))

        fig.write_image(filename)
        fig.show()

        
    def generate_video(self, filename):
        self.env.video(self.policy, filename=filename)
    
    def plot_trajectory(self, filename1, filename2):
        s = self.env.reset()

        data = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
        }

        done = False
        while not done:
            a = self.policy(s)
            (s, r, done) = self.env.step(a)
            data['t'].append(data['t'][-1] + 1)
            data['s'].append(s)
            data['a'].append(a)
            data['r'].append(r)

        data['s'] = np.array(data['s'])
        theta = data['s'][:, 0]
        thetadot = data['s'][:, 1]
        tau = [self.env._a_to_u(a) for a in data['a']]

        pio.templates.default = "plotly_dark"

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        fig.add_trace(
            go.Scatter(x=data['t'], y=theta, name=r'$\theta$'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['t'], y=thetadot, name=r'$\dot \theta$'),
            row=1, col=1
        )
        fig.update_xaxes(title_text="Time step", row=3, col=1)
        fig.update_yaxes(title_text="Angle (rad)", row=1, col=1)
        fig.update_yaxes(title_text="Angular velocity (rad/s)", row=1, col=1)
        fig.update_layout(title="Pendulum Trajectory",
                        showlegend=True,
                        legend=dict(font=dict(size=12)),
                        height=600,
                        margin=dict(l=50, r=50, t=80, b=50))

        fig.add_trace(
            go.Scatter(x=data['t'][:-1], y=tau, name=r'$\tau$'),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Time step", row=3, col=1)
        fig.update_yaxes(title_text="Torque (N-m)", row=2, col=1)
        fig.update_layout(title="Control Signal",
                        showlegend=True,
                        legend=dict(font=dict(size=12)),
                        height=400,
                        margin=dict(l=50, r=50, t=50, b=50))


        fig.add_trace(
            go.Scatter(x=data['t'][:-1], y=data['r'], name='Reward'),
            row=3, col=1
        )
        fig.update_xaxes(title_text="Time step", row=3, col=1)
        fig.update_yaxes(title_text="Reward", row=3, col=1)
        fig.update_layout(title="Reward Function",
                        showlegend=True,
                        legend=dict(font=dict(size=12)),
                        height=400,
                        margin=dict(l=50, r=50, t=50, b=50))

        fig.write_image(filename1)
        fig.show()


    def plot_ablation_study(self, gamma, file_list, legend_list, dest_file):
        num_files = len(file_list)
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.1)

        for i in range(num_files):
            source_file = file_list[i]
            with open(source_file) as f:
                lines = f.readlines()

            lines = [line.strip() for line in lines]

            episode_list = [[int(num) for num in line[1:-1].split(',')] 
                            for line in lines]
            num_eps              = len(episode_list)
            discounted_returns   = np.zeros(num_eps)
            undiscounted_returns = np.zeros(num_eps)
            for e in range(num_eps):
                returns   = 0
                u_returns = 0
                for j in range(len(episode_list[e])):
                    returns   += (gamma**j) * episode_list[e][j]
                    u_returns += episode_list[e][j]
                discounted_returns[e]   = returns
                undiscounted_returns[e] = u_returns
            
            mean_ud_returns, sd_ud_returns = [], []
            mean_d_returns, sd_d_returns = [], []

            for j in range(num_eps):
                if j < 100:
                    mean_ud_returns.append(np.mean(undiscounted_returns[0:j+1]))
                    sd_ud_returns.append(np.std(undiscounted_returns[0:j+1]))
                    mean_d_returns.append(np.mean(discounted_returns[0:j+1]))
                    sd_d_returns.append(np.std(discounted_returns[0:j+1]))
                else:
                    mean_ud_returns.append(np.mean(undiscounted_returns[j-100:j]))
                    sd_ud_returns.append(np.std(undiscounted_returns[j-100:j]))
                    mean_d_returns.append(np.mean(discounted_returns[j-100:j]))
                    sd_d_returns.append(np.std(discounted_returns[j-100:j]))

            row_means = np.array(mean_d_returns)
            row_stds = np.array(sd_d_returns)
            df_d = pd.DataFrame({'x': range(num_eps),
                                'y': row_means,
                                'lower': row_means - row_stds,
                                'upper': row_means + row_stds})

            fig.add_trace(go.Scatter(x=df_d['x'], y=df_d['y'], name=legend_list[i], mode='lines'),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=df_d['x'], y=df_d['lower'], name='', showlegend=False, mode='lines'),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=df_d['x'], y=df_d['upper'], name='', showlegend=False, fill='tonexty', mode='lines'),
                        row=1, col=1)

            row_means = np.array(mean_ud_returns)
            row_stds = np.array(sd_ud_returns)
            df_ud = pd.DataFrame({'x': range(num_eps),
                                'y': row_means,
                                'lower': row_means - row_stds,
                                'upper': row_means + row_stds})

            fig.add_trace(go.Scatter(x=df_ud['x'], y=df_ud['y'], name=legend_list[i], mode='lines'),
                        row=2, col=1)
            fig.add_trace(go.Scatter(x=df_ud['x'], y=df_ud['lower'], name='', showlegend=False, mode='lines'),
                        row=2, col=1)
            fig.add_trace(go.Scatter(x=df_ud['x'], y=df_ud['upper'], name='', showlegend=False, fill='tonexty', mode='lines'),
                        row=2, col=1)

        fig.update_layout(title='DQN Learning Curve', height=800)
        fig.update_xaxes(title_text='Episodes', row=2, col=1)
        fig.update_yaxes(title_text='Mean Discounted Returns', range=[-15, 20], row=1, col=1)
        fig.update_yaxes(title_text='Mean Undiscounted Returns', range=[-110, 110], row=2, col=1)
        fig.write_image(dest_file)
        fig.show()
        
    def plot_ablation_study_bar(self, num_eps, gamma, weight_list, dest_file):
        
        x_list          = ['with target, with replay', 
                          'with target, without replay',
                          'without target, with replay', 
                          'without target, without replay']
        max_mean_ud     = []
        max_mean_d      = []
        w_size          = 100
        for weight_file in weight_list:
            self.NN.load_state_dict(torch.load(weight_file))
            un_returns_list = []
            d_returns_list  = []
            for eps in range(num_eps):
                done       = False
                s          = self.env.reset()
                d_returns  = 0.0
                ud_returns = 0.0
                count      = 0
                while not done:
                    a             = self.policy(s)
                    (s, r, done)  = self.env.step(a)
                    d_returns    += (gamma ** count) * r
                    ud_returns   += r
                    count        += 1 
                un_returns_list.append(ud_returns)
                d_returns_list.append(d_returns)
                
            max_mean_ud.append(np.max(np.convolve(un_returns_list, 
                                           np.ones(w_size)/w_size, 
                                           mode='valid')))
            max_mean_d.append(np.max(np.convolve(d_returns_list, 
                                           np.ones(w_size)/w_size, 
                                           mode='valid')))
            
                
                    
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Mean Discounted Returns", "Mean Undiscounted Returns"))

        bar1 = go.Bar(x=x_list, y=max_mean_d, name="Mean Discounted Returns")

        bar2 = go.Bar(x=x_list, y=max_mean_ud, name="Mean Undiscounted Returns")

        fig.add_trace(bar1, row=1, col=1)
        fig.add_trace(bar2, row=2, col=1)

        fig.update_yaxes(title_text="Mean Discounted Returns", row=1, col=1)
        fig.update_yaxes(title_text="Mean Undiscounted Returns", row=2, col=1)
        fig.update_xaxes(title_text="Episodes", row=2, col=1)

        fig.update_layout(title="Maximum mean episode reward", height=600, width=800, showlegend=True)
        fig.update_layout(margin=dict(l=50, r=50, b=50, t=70, pad=0))

        fig.write_image(dest_file)
        fig.show()
        
