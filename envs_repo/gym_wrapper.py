import numpy as np
import gym
import pickle
import random

def standardScale(aa):
    K = 3.29 #99.9%
    #K = 1.96 #95%
    meanV = np.mean(aa)
    stdV = np.std(aa)
    #pp_meanV = 92.028267 #train_CZF
    #pp_stdV = 50.173733 #train_CZF
    #ss_meanV = 0.0014739 #train_CCF
    #ss_stdV = 0.0240361 #train_CCF
    #pp_meanV = 49.578292 #train_CCF
    #pp_stdV = 10.992933 #train_CCF
    x_std = (aa-meanV)/(K*stdV)
  
    return x_std, meanV, stdV  


#with open('/content/Evolutionary-Reinforcement-StockDayTrade/envs_repo/train_CZF.pickle','rb') as f:
with open('/content/Evolutionary-Reinforcement-StockDayTrade/envs_repo/train_CCF.pickle','rb') as f:
    data = pickle.load(file = f)
l = 284 #13:29=> the last one
dataN = len(data)
ss = np.zeros((l,dataN))
pp = np.zeros((l,dataN))
symBe = ''
base = 0

for i in range(dataN):
    sym = (data[i][1])[3:5] #20221109
    if sym != symBe:

        #for s in range(l-1):
        for s in range(l):
            #ss[s,i] = (data[i][0].iloc[s,1]/data[i][0].iloc[0,1]-1.)
            ss[s,i] = np.log(data[i][0].iloc[s,1])-np.log(data[i][0].iloc[0,1])
            pp[s,i] = data[i][0].iloc[s,1]
        symBe = sym
        base = data[i][0].iloc[l-1,1]

    else:
        #for s in range(l-1):
        for s in range(l):
            #ss[s,i] = (data[i][0].iloc[s,1]/base-1.)
            ss[s,i] = np.log(data[i][0].iloc[s,1])-np.log(base)
            pp[s,i] = data[i][0].iloc[s,1]
        symBe = sym
        base = data[i][0].iloc[l-1,1]  

############## train_CZF #####################
#ss, _, _ = standardScale(ss)#20221111 僅pp作正規化
#pp, _, _ = standardScale(pp)
###########################################
############## train_CCF #####################
ss, _, _ = standardScale(ss)#max: 1.235149561798326  min: -1.2513307744776638
pp, _, _ = standardScale(pp)
################################################
#print('maxmaxmaxmax  ',np.max(ss),' minminminmin ',np.min(ss))
window_size = 4
tradeCost = 47
   
class GymWrapper(gym.Env):

  def __init__(self, env_name, frameskip=None):
      self.state_dim = 6
      self.action_dim = 3
      self.pIndex = ss
      self.price_std = pp
      self.price = data
      self.commission = 47
      self.position = np.array([0.])
      self.inventory = []
      self.d = 0 # day
      self.t = 0 # time
      self.st = 0 # tradin time
      self.done = False

  def is_discrete(self, env):
        try:
            k = env.action_space.n
            return True
        except:
            return False

  def getStateTv(self):
      aa = self.pIndex[self.t-4:self.t , self.d]
      pri_s = np.array([self.price_std[self.t-1, self.d]])
      aa = np.concatenate((aa, pri_s, self.position), axis=0)
      return aa

  def _take_action(self, action):
      reward = 0
      if action == 1:
          if int(self.position[0]) == 0:
              self.position = np.array([1.])
              self.inventory.append(self.price[self.d][0].iloc[self.t-1,1])
              self.st = self.t # tradin time

          if int(self.position[0]) == -1:
              sold_price = self.inventory.pop(0)
              reward = 2000*(sold_price - self.price[self.d][0].iloc[self.t-1,1])-2*self.commission
              self.done = True
              self.position = np.array([0.])
      elif action == 2:
          if int(self.position[0]) == 0:
              self.position = np.array([-1.])
              self.inventory.append(self.price[self.d][0].iloc[self.t-1,1])
              self.st = self.t # tradin time

          if int(self.position[0]) == 1:
              bought_price = self.inventory.pop(0)
              reward = 2000*(self.price[self.d][0].iloc[self.t-1,1] - bought_price)-2*self.commission
              self.done = True
              self.position = np.array([0.])
       
      return reward, self.position

  def step(self, action):
      reward, self.position = self._take_action(action)
      self.t += 1
      #observation = self.getState()
      observation = self.getStateTv()
      if self.t == 284:
          self.done = True
          if len(self.inventory) > 0:
              if int(self.position[0]) == 1:
                  bought_price = self.inventory.pop(0)
                  reward = 2000*(self.price[self.d][0].iloc[self.t-1,1] - bought_price)-2*self.commission
                  #observation[self.state_dim+1] = np.array([0.])
                  observation[5] = np.array([0.])#20220509

              elif int(self.position[0]) == -1:
                  sold_price = self.inventory.pop(0)
                  reward = 2000*(sold_price - self.price[self.d][0].iloc[self.t-1,1])-2*self.commission
                  #observation[self.state_dim+1] = np.array([0.])
                  observation[5] = np.array([0.])#20220509
      info = {'env.d':self.d, 'env.t':self.t, 'reward':reward, 'tradein t':self.st}
      #print('observation  ',observation)
      #print('self.done  ',self.done)
      return observation, reward, self.done, info

  def reset(self):
      self.position = np.array([0.])
      self.inventory = []
      dth = random.randint(0, self.pIndex.shape[1]-1)
      self.d = dth # day
      self.t = 4 # time
      self.st = 4 # tradin time
      self.done = False

      return self.getStateTv() 

  def render(self):
      pass
