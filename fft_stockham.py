import numpy as np #purely as test, cmath outperform np.exp
import cmath

def twiddle(k,n,N):
    return cmath.exp(-2j * cmath.pi * k * n / N) #Simplifying the twiddle factor W(p,N)

def puredft(x): #DFT of x, i.e output=DFT(x)
    N=len(x)
    y=[0]*len(x)
    for a in range(N):
        c0=0
        for b in range(N):
            c0+=x[b]*twiddle(a,b,N)
        y[a]=c0
    return y

def logn(n,x): # return integers of a log i.e log2(8)=3, log2(6)=error, n is the base of the log
    assert x%n==0
    i=1
    if x<n:
        return 0
    while x/n !=1:
        i=i+1
        x/=n
        assert x%n==0
    return i

def rad2fft(arr): 
    N=len(arr)
    p=logn(2,N)
    l=N//2
    m=1
    j=(-1)**0.5
    scratch=arr[:]
    for t in range(1,p+1):
        for i in range(0,l):
            wq= cmath.exp(-2j*cmath.pi*i/(2*l))
            for k in range(0,m):
                c0=arr[k+i*m]
                c1=arr[k+i*m+l*m]
                
                scratch[k+2*i*m]=c0+c1
                scratch[k+2*i*m+m]=wq*(c0-c1)
        l=l//2
        m=m*2
        arr=scratch[:]
    return arr

def rad4fft(arr):
    N=len(arr)
    p = logn(4,N)
    l=N//4
    m=1
    j=(-1)**0.5
    scratch=arr[:]
    for t in range(1,p+1):
        for i in range(0,l):
            wq =cmath.exp(-2j*cmath.pi*i/(4*l))
            for k in range(0,m):
                c0 = arr[k+i*m]
                c1 = arr[k+i*m+l*m]
                c2 = arr[k+i*m+2*l*m]
                c3 = arr[k+i*m+3*l*m]
                d0=c0+c2
                d1=c1+c3
                d2=c0-c2
                d3=-j*(c1-c3)
                
                scratch[k+4*i*m]     = d0+d1
                scratch[k+4*i*m+m]   = wq*(d2+d3)
                scratch[k+4*i*m+2*m] = (wq**2)*(d0-d1)
                scratch[k+4*i*m+3*m] = (wq**3)*(d2-d3)
        l=l//4
        m=m*4
        arr=scratch[:]
    return arr

def rad16fft(arr):
    N=len(arr)
    p = logn(16,N)
    l=N//16
    m=1
    j=(-1)**0.5
    w1=cmath.exp(-2j*cmath.pi/16) 
    w2=(2**0.5)/2-j*(2**0.5)/2
    w3=cmath.exp(-6j*cmath.pi/16)
    w4=-j
    w5=cmath.exp(-10j*cmath.pi/16)
    w6=-(2**0.5)/2-j*(2**0.5)/2
    w7=cmath.exp(-14j*cmath.pi/16)
    scratch=arr[:]
    for t in range(1,p+1):
        for i in range(0,l):
            wq = cmath.exp(-2j*cmath.pi*i/(16*l))
            for k in range(0,m):
                
                c0 = arr[k+i*m]
                c1 = arr[k+i*m+l*m]
                c2 = arr[k+i*m+2*l*m]
                c3 = arr[k+i*m+3*l*m]
                c4 = arr[k+i*m+4*l*m]
                c5 = arr[k+i*m+5*l*m]
                c6 = arr[k+i*m+6*l*m]
                c7 = arr[k+i*m+7*l*m]
                c8 = arr[k+i*m+8*l*m]
                c9 = arr[k+i*m+9*l*m]
                c10= arr[k+i*m+10*l*m]
                c11= arr[k+i*m+11*l*m] 
                c12= arr[k+i*m+12*l*m]
                c13= arr[k+i*m+13*l*m]
                c14= arr[k+i*m+14*l*m]
                c15= arr[k+i*m+15*l*m]
                
                e0=c0+c8
                e1=c1+c9
                e2=c2+c10
                e3=c3+c11
                e4=c4+c12
                e5=c5+c13
                e6=c6+c14
                e7=c7+c15
                e8=c0-c8
                e9=c1-c9
                e10=c2-c10
                e11=c3-c11
                e12=c4-c12
                e13=c5-c13
                e14=c6-c14
                e15=c7-c15
                
                f0=e0+e4
                f1=e0-e4
                f2=e2+e6
                f3=e2-e6
                f4=e1+e5
                f5=e1-e5
                f6=e3+e7
                f7=e3-e7
                f8=e8+w4*e12
                f9=e8-w4*e12
                f10=w2*e10+w6*e14
                f11=w6*e10+w2*e14
                f12=e9-j*e13
                f13=e11-j*e15
                f14=e9+j*e13
                f15=e11+j*e15
                
                d0= f0+f2
                d1= f4+f6
                d2= f8+f10
                d3= w1*f12+w3*f13
                d4= f1+w4*f3
                d5= w2*f5+w6*f7
                d6= f9+f11
                d7= w3*f14-w1*f15
                d8= f0-f2
                d9= w4*(f4-f6)
                d10=f8-f10
                d11=w5*f12-w7*f13
                d12=f1-w4*f3
                d13=w2*f7+w6*f5
                d14=f9-f11
                d15=w7*f14+w5*f15
                
                scratch[k+16*i*m]     = d0+d1
                scratch[k+16*i*m+m]   = wq*(d2+d3)
                scratch[k+16*i*m+2*m] = (wq**2)*(d4+d5)
                scratch[k+16*i*m+3*m] = (wq**3)*(d6+d7)
                scratch[k+16*i*m+4*m] = (wq**4)*(d8+d9)
                scratch[k+16*i*m+5*m] = (wq**5)*(d10+d11)
                scratch[k+16*i*m+6*m] = (wq**6)*(d12+d13)
                scratch[k+16*i*m+7*m] = (wq**7)*(d14+d15)
                scratch[k+16*i*m+8*m] = (wq**8)*(d0-d1)
                scratch[k+16*i*m+9*m] = (wq**9)*(d2-d3)
                scratch[k+16*i*m+10*m] = (wq**10)*(d4-d5)
                scratch[k+16*i*m+11*m] = (wq**11)*(d6-d7)
                scratch[k+16*i*m+12*m] = (wq**12)*(d8-d9)
                scratch[k+16*i*m+13*m] = (wq**13)*(d10-d11)
                scratch[k+16*i*m+14*m] = (wq**14)*(d12-d13)
                scratch[k+16*i*m+15*m] = (wq**15)*(d14-d15)
        l=l//16
        m=m*16
        arr=scratch[:]
    return arr