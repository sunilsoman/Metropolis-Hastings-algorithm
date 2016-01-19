import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import multivariate_normal

#number of sample points
n = 1000 

def read_data(filepath, d = ','):
    """ returns an np.array of the data """
    return np.genfromtxt(filepath, delimiter=d, dtype=None)

def plot_data(r, qi,qf,piArray,pfArray,filename):
    '''
    plt.figure()
    plt.plot(r[:,0],r[:,1])
    plt.scatter(qi[0],qi[1],edgecolor='r', color='y')
    plt.scatter(qf[0],qf[1],edgecolor='g', color='y')
    '''
    s = np.linspace(0,n,n+1)
    plt.figure()
    plt.suptitle('3d points pi,pf vs sample points ('+filename+')')
    plt.subplots_adjust(hspace=0.7)
    plt.subplot(3,2,1)
    plt.plot(s,piArray[0:len(piArray):3])
    plt.xlabel('Number of sample points')
    plt.ylabel('pi_x')
    plt.title('pi_x Vs Sample points')
    plt.subplot(3,2,3)
    plt.plot(s,piArray[1:len(piArray):3])
    plt.xlabel('Number of sample points')
    plt.ylabel('pi_y')
    plt.title('pi_y Vs Sample points')
    plt.subplot(3,2,5)
    plt.plot(s,piArray[2:len(piArray):3])
    plt.xlabel('Number of sample points')
    plt.ylabel('pi_z')
    plt.title('pi_z Vs Sample points')
    plt.subplot(3,2,2)
    plt.plot(s,pfArray[0:len(pfArray):3])
    plt.xlabel('Number of sample points')
    plt.ylabel('pf_x')
    plt.title('pf_x Vs Sample points')
    plt.subplot(3,2,4)
    plt.plot(s,pfArray[1:len(pfArray):3])
    plt.xlabel('Number of sample points')
    plt.ylabel('pf_y')
    plt.title('pf_y Vs Sample points')
    plt.subplot(3,2,6)
    plt.plot(s,pfArray[2:len(pfArray):3])
    plt.xlabel('Number of sample points')
    plt.ylabel('pf_z')
    plt.title('pf_z Vs Sample points')
    plt.show()

def run_MH(input_datapath, points_datapath, M, filename, calEstimate=False):
    """function uses Metropolis- Hastings Algorithm to find MAP estimate of 3D line segment"""

    #input data
    t = read_data(input_datapath, ',')

    #2D data points
    r = read_data(points_datapath, ',')

    #mean and variance for prior
    mu = np.array([0,0,4])
    var = 10*np.identity(3)

    #variance for likelihood
    co_var = 0.0025*np.identity(2)

    #random initial and final 3D points 
    pi,pf = np.random.multivariate_normal(mu, var, 2)

    #initializing corresponding initial & final 2D points
    qi,qf = [0]*2
    '''
    piArray =np.empty([n+1,3])
    pfArray =np.empty([n+1,3])
    piArray =np.array(piArray,pi)
    pfArray =np.append(pfArray,pf)
    '''
    piArray=np.array(pi)
    pfArray=np.array(pf)

    for i in range(0,n):
        pa=np.append(pi, 1)
        pb=np.append(pf, 1)

        xi = M.dot(pa)
        xf = M.dot(pb)

        xi = xi/xi[2]
        xf = xf/xf[2]

        qi = xi[0:2]
        qf = xf[0:2]

        qs =np.zeros((t.shape[0],2)) 

        #Calculating qs which is the mean for likelihood
        for k in range(0,t.shape[0]):
            qs[k]=np.add(qi, np.subtract(qf,qi)*t[k])

        ##Calculating posterior for old points
        prior = multivariate_normal.logpdf([pi,pf], mean=mu, cov=var)
        likelihood = np.zeros((t.shape[0],1))

        for j in range(0, t.shape[0]):
            #likelihood[j] =multivariate_normal.logpdf(r[j,:], mean=qs[j,:], cov=co_var)
            likelihood[j] =multivariate_normal.logpdf(r[j], mean=qs[j], cov=co_var)
        
        #p_old - posterior for old points
        p_old = np.sum(prior)+np.sum(likelihood)

        ##new points
        pi_new= np.random.multivariate_normal(pi,var)
        pf_new= np.random.multivariate_normal(pf,var)

        #pi_new +=pi
        #pf_new +=pf
        piArray =np.append(piArray,pi_new)
        pfArray =np.append(pfArray,pf_new)

        pa=np.append(pi_new, 1)
        pb=np.append(pf_new, 1)

        xi = M.dot(pa)
        xf = M.dot(pb)

        xi = xi/xi[2]
        xf = xf/xf[2]

        qi_new = xi[0:2]
        qf_new = xf[0:2]

        qs =np.zeros((t.shape[0],2)) 

        for k in range(0,t.shape[0]):
            qs[k]=np.add(qi_new, np.subtract(qf_new,qi_new)*t[k])

        #Calculating posterior for new points
        
        prior = multivariate_normal.logpdf([pi_new,pf_new], mean=mu, cov=var)
        likelihood = np.zeros((t.shape[0],1))

        for j in range(0, t.shape[0]):
            #likelihood[j] =multivariate_normal.logpdf(r[j,:], mean=qs[j,:], cov=co_var)
            likelihood[j] =multivariate_normal.logpdf(r[j], mean=qs[j], cov=co_var)
        

        #p_new - posterior for new points
        p_new = np.sum(prior)+np.sum(likelihood)
        
        if(p_new-p_old>=0):
            pi=pi_new
            pf=pf_new
            qi=qi_new
            qf=qf_new
        else:
            u = np.random.uniform(0.0,1.0,1)
            if(np.log(u)<p_new-p_old):
                pi=pi_new
                pf=pf_new
                qi=qi_new
                qf=qf_new

    print '\n----------------------\nResults ('+filename+')'
    print  '\nMAP estimate of the 3D line segment (pi&pf):{0},{1}'.format(pi,pf)
    print  '\nCorresponding 2D points (qi&qf):{0},{1}'.format(qi,qf)

    if(calEstimate):
        print '\n Mean of posterior distribution (E[pi],E[pf]):{0},{1}'.format(pi,pf)
        qs = qi+(qf-qi)*1.5
        print '\n Monte Carlo estimate of the predicted 2D output point:{0}'.format(qs)
    
    plot_data(r, qi,qf,piArray,pfArray,filename)

#file locations for input, points_2d, points_2d_2
input_datapath = 'project_option_A_data_release/inputs.csv'
points_2d_datapath = 'project_option_A_data_release/points_2d.csv'
points_2d_2_datapath = 'project_option_A_data_release/points_2d_2.csv'

#initializing the Camera matrix
M = np.hstack((np.identity(3), np.atleast_2d(np.array([0,0,0])).T))
M2= np.array([[0,0,1,-5], [0,1,0,0], [-1,0,0,5]])

run_MH(input_datapath, points_2d_datapath, M,'points2d.csv',True)
run_MH(input_datapath, points_2d_2_datapath, M2,'points_2d_2.csv')
