import numpy as np
import csv

class NeuralNetwork:
    def __init__(self):
        #number of data in one forward pass
        self.no_data = 100
        #number of traing iterations
        self.no_iterations = 5000
        self.learning_rate = 1
        np.random.seed(0)
        #number of nodes in layers(input, hidden, output)
        self.no_inputs      = 1
        self.no_ht          = 5
        self.no_outputs     = 1
        self.no_inputs_total= self.no_inputs + self.no_ht
        #np.random.seed(0)
        #initializing layers and weights
        self.input          = np.zeros((self.no_data + 1,self.no_inputs))
        self.output         = np.zeros((self.no_data + 1,self.no_outputs))
        self.ht             = np.zeros((self.no_data + 1, self.no_ht))
        self.ct             = np.zeros((self.no_data + 1, self.no_ht))
        self.w2             = np.random.rand(self.no_ht, self.no_outputs)
        #self.output_target  = np.zeros((self.no_data + 1,self.no_outputs))
        #initializing lstm gates(input, forget, output, gate gate)
        self.ig             = np.zeros((self.no_data + 1, self.no_ht))
        self.fg             = np.zeros((self.no_data + 1, self.no_ht))
        self.og             = np.zeros((self.no_data + 1, self.no_ht))
        self.gg             = np.zeros((self.no_data + 1, self.no_ht))
        
        #initializing lstm weights
        self.wi             = np.random.rand(self.no_inputs_total,self.no_ht)
        self.wf             = np.random.rand(self.no_inputs_total,self.no_ht)
        self.wo             = np.random.rand(self.no_inputs_total,self.no_ht)
        self.wg             = np.random.rand(self.no_inputs_total,self.no_ht)

        #initializing biases
        self.bf = np.random.rand(self.no_data + 1,self.no_ht) 
        self.bi = np.random.rand(self.no_data + 1,self.no_ht)
        self.bg = np.random.rand(self.no_data + 1,self.no_ht)
        self.bo = np.random.rand(self.no_data + 1,self.no_ht)
        self.by = np.random.rand(self.no_data + 1,self.no_outputs)

    #activation function
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    # convert output of sigmoid function to its derivative
    def sigmoid_derivative(self,x):
        return x*(1-x)
    
    #conver outout of tang function to its derivative
    def tanh_derivative(self, x):
        return 1 - x**2
    
    #function that imports data from csv file and puts it in input vector
    def import_data(self,file):
        with open(file,'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            i = 1
            for arg_list in csv_reader:
                
                x = list(np.float_(arg_list))
                x = x[1:]
                self.input[i] = np.array(x, ndmin = 2)
                i = i + 1
                if(i >= self.no_data ):
                    break
    
    #calculating ouptu and rolling out nework for number of data in our data set
    def feedforward(self):

        for i in range(1,self.no_data + 1):
            
            #concatenating new_input + hidden
            input_total = np.atleast_2d(np.concatenate([self.input[i],self.ht[i-1]]))
            #forget gate
            self.fg[i]  = self.sigmoid(input_total @ self.wf + self.bf[i])
            #input gate
            self.ig[i]  = self.sigmoid(input_total @ self.wi + self.bi[i] )
            #output gate
            self.og[i]  = self.sigmoid(input_total @ self.wo + self.bo[i])
            #gate gate
            self.gg[i]  = np.tanh(input_total @ self.wi + self.bg[i])
            #cell state
            self.ct[i]  = (self.fg[i] * self.ct[i-1]) + ( self.ig[i] * self.gg[i])
            #hidden layer output
            self.ht[i]  = self.og[i]  * np.tanh(self.ct[i])
            #network output
            self.output[i] = self.sigmoid(self.ht[i] @ self.w2 + self.by[i])
        
        

    def backprop(self):
        
        #initialize derivative of hidden layer and cell state 
        d_ht = np.zeros(self.no_ht)
        d_ct = np.zeros(self.no_ht)

        #change in w2 and bias
        delta_w2 = np.zeros((self.no_ht,self.no_outputs))
        delta_by = np.zeros((self.no_data + 1,self.no_outputs))

        #change in bias for lstm cell
        delta_bf = np.zeros((self.no_data + 1,self.no_ht)) 
        delta_bi = np.zeros((self.no_data + 1,self.no_ht))
        delta_bg = np.zeros((self.no_data + 1,self.no_ht))
        delta_bo = np.zeros((self.no_data + 1,self.no_ht))

        #forget gate
        sum_delta_f = np.zeros(( self.no_inputs+self.no_ht,self.no_ht))
        #input gate
        sum_delta_i = np.zeros((self.no_inputs+self.no_ht,self.no_ht))
        #gate gate/ cell unit
        sum_delta_g = np.zeros((self.no_inputs+self.no_ht,self.no_ht))
        #output gate
        sum_delta_o = np.zeros((self.no_inputs+self.no_ht,self.no_ht))
        err_tot = 0
        for i in range(self.no_data - 1, -1, -1):
            #calculating error fot this iteration
            err = np.atleast_2d(self.output[i]- self.input[i+1])

            #change for weights  and bias for output layer
            delta_w2 += np.atleast_2d(self.ht[i]).T @ (err * self.sigmoid_derivative(self.output[i]))
            delta_by += err * self.sigmoid_derivative(self.output[i])
            
            #porpagating error backt to LSTM
            err = err @ self.w2.T

            #clipping for expling gradient
            err = np.clip(err + d_ht, -6, 6)
            total_input = np.atleast_2d(np.concatenate([self.input[i],self.ht[i-1]]))
            
            #change in output gate and bias
            d_o = err * np.tanh(self.ct[i])
            delta_o = total_input.T @ (d_o * self.sigmoid_derivative(self.og[i]))
            delta_bo += (d_o * self.sigmoid_derivative(self.og[i])) 
            
            #derivative for cell 
            d_cs = np.clip(err * self.og[i] * self.tanh_derivative(np.tanh(self.ct[i])) + d_ct,-6,6)

            #change in gate gate and bias
            d_g = d_cs * self.ig[i]
            delta_g = total_input.T @ (d_g * self.tanh_derivative(self.gg[i]))
            delta_bg += (d_g * self.tanh_derivative(self.gg[i]))

            #change in input gate and bias
            d_i = d_cs * self.gg[i]
            delta_i = total_input.T @ (d_i * self.sigmoid_derivative(self.ig[i]))
            delta_bi += (d_i * self.sigmoid_derivative(self.ig[i]))

            #change in forget gate and bias
            d_f = d_cs * self.ct[i-1]
            delta_f = total_input.T @ (d_f * self.sigmoid_derivative(self.fg[i]))
            delta_bf += (d_f * self.sigmoid_derivative(self.fg[i]))

            #derivative cell and hidden
            d_ct = d_cs * self.fg[i]
            d_ht = (d_g @ self.gg[i])[:self.no_ht] + (d_o @ self.og[i])[:self.no_ht] + (d_i @ self.ig[i])[:self.no_ht] + (d_f @ self.fg[i])[:self.no_ht] 
            
            #sum gradients for update for all 4 gates
            sum_delta_f += delta_f
            #input gate
            sum_delta_i += delta_i
            #cell state
            sum_delta_g += delta_g
            #output gate
            sum_delta_o += delta_o

        # Alterantive formulas
        # Gf = (sum_delta_f/self.no_data)**2 
        # Gi = (sum_delta_i/self.no_data)**2   
        # Gg = (sum_delta_g/self.no_data)**2   
        # Go = (sum_delta_o/self.no_data)**2  
        
        # #Update our gates using our gradients
        # self.wf -= self.learning_rate/np.sqrt(Gf + 1e-8) * (sum_delta_f/self.no_data)
        # self.wi -= self.learning_rate/np.sqrt(Gi + 1e-8) * (sum_delta_i/self.no_data)
        # self.wg -= self.learning_rate/np.sqrt(Gg + 1e-8) * (sum_delta_g/self.no_data)
        # self.wo -= self.learning_rate/np.sqrt(Go + 1e-8) * (sum_delta_o/self.no_data)

        # Gy = delta_w2**2  
        # self.w2 -= self.learning_rate/np.sqrt(Gy + 1e-8) * (delta_w2/self.no_data)
        # Bf = (delta_bf/self.no_data)**2 
        # Bi = (delta_bi/self.no_data)**2   
        # Bg = (delta_bg/self.no_data)**2   
        # Bo = (delta_bo/self.no_data)**2  

        # self.bf -= self.learning_rate/np.sqrt(Bf + 1e-8) * (delta_bf/self.no_data)
        # self.bi -= self.learning_rate/np.sqrt(Bi + 1e-8) * (delta_bi/self.no_data)
        # self.bg -= self.learning_rate/np.sqrt(Bg + 1e-8) * (delta_bg/self.no_data)
        # self.bo -= self.learning_rate/np.sqrt(Bo + 1e-8) * (delta_bo/self.no_data)
        
        # By = delta_by**2  
        # self.by -= self.learning_rate/np.sqrt(By + 1e-8) * (delta_by/self.no_data)

        self.wf -= self.learning_rate * sum_delta_f/self.no_data
        self.wi -= self.learning_rate * sum_delta_i/self.no_data
        self.wg -= self.learning_rate * sum_delta_g/self.no_data
        self.wo -= self.learning_rate * sum_delta_o/self.no_data
       
        self.w2 -= self.learning_rate * delta_w2/self.no_data

        self.bf -=  self.learning_rate * (delta_bf/self.no_data)
        self.bi -=  self.learning_rate * (delta_bi/self.no_data)
        self.bg -=  self.learning_rate * (delta_bg/self.no_data)
        self.bo -=  self.learning_rate * (delta_bo/self.no_data)


    #function for training
    def train(self):
            
        for i in range(self.no_iterations):
            if(i%100 == 0):
                print(i)    
            self.feedforward()
            self.backprop()

    #final evaluation function where we see results of out network
    def evaluate(self):
        self.feedforward()
        # for i in range(self.no_data):
        #     print(self.output[i])
        x =  list(np.arange(-np.pi,np.pi, np.pi/45))
        y =  list(self.output)
    
    #function that calculates percentage of success
    def perc(self):
        sum1 = 0
        for i in range(1,self.no_data-1):
            if(np.abs(self.input[i] - self.output[i]) <= 0.5 ):
                sum1 += 1
        
        print(sum1/(self.no_data-2.0) * 100.0)
        

        

if __name__ == "__main__":

    neural_network = NeuralNetwork()
    neural_network.import_data('/home/filip/1_Project_RI/data.csv')
    neural_network.train()
    neural_network.evaluate()
    neural_network.perc()
