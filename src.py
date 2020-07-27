import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from collections.abc import Iterable
%matplotlib notebook
D_='<location of data>'
DA_=D_+'analysis\\'
A1='DC:A6:32:33:B0:E6'
A2='DC:A6:32:33:AF:9B'
F1T12=[i*12 for i in range(1,13)]
def dist(file):
    return int(file[file.find('D')+1:file.find('@')])
def expi(file):
    return int(file[1:file.find('D')])
def rssis(file,**kwargs):
    adver=file[file.find('@')+1:]
    if adver!='1' and adver!='2':
        adver=str(kwargs['adver'])
    df=pd.read_csv(D_+file+'.csv')
    return list(df[df['ADDRESS']==(A1 if adver=='1' else A2)]['RSSI'])
def files(e,ds):
    return ['E'+str(e)+'D'+str(d)+'@'+str(a) for d in ds for a in [1,2]]
def files12(e,ds):
    return ['E'+str(e)+'D'+str(d)+'@1,2' for d in ds]
FSS=[
    None,
    files(1,[*[i*12 for i in range(1,9)],120]),
    files(2,[*[i*12 for i in range(1,16)],12*17,12*20,12*25]),
    files(3,F1T12),
    files(4,[i*12 for i in range(2,11)]),
    *[files(i,F1T12) for i in range(5,9)],
    *[files12(i,F1T12) for i in [9,10]],
]
def avgs(ax,files,**kwargs):
    cmap=plt.cm.get_cmap('tab20')
    i=0
    k1={**kwargs}
    k1.pop('adver',None)
    for f in files:
        r=rssis(f,**kwargs)
        #print(f+':'+str(np.mean(r)))
        ax.scatter(dist(f),np.mean(r),label=f,color=cmap(i/20),**k1)
        i+=1
def boxes(ax,files,**kwargs):
    cmap=plt.cm.get_cmap('tab20')
    data=[]
    dists=[]
    for f in files:
        data.append(rssis(f,**kwargs))
        dists.append(dist(f))
    kwargs.pop('adver',None)
    ax.boxplot(data,positions=dists,widths=[6 for f in files],**kwargs)
def lines(ax,files,**kwargs):
    for f in files:
        r=rssis(f,**kwargs)
        ax.plot(r,label=' '+f)
        #print('f='+f+', len='+str(len(r))+', avg='+str(np.mean(r)))
    ax.legend()
def hists(files,legend,**kwargs):
    fig, ax=plt.subplots()
    print('avgs:')
    cmap=plt.cm.get_cmap('tab20')
    i=0
    k1={**kwargs}
    k1.pop('adver',None)
    for f in files:
        r=rssis(f,**kwargs)
        print(f+':'+str(np.mean(r)))
        ax.hist(r,label=' '+f,color=cmap(i/20),bins=np.max(r)-np.min(r)+1,**k1)
        i+=1
    if legend:
        ax.legend()
    ax.set_title('histogram of RSSIs of experiment #1')
    ax.set_xlabel('RSSIs')
    plt.show()
    plt.save_fig(DA_+'tmp.png')
def data(files,**kwargs):
    ds=[]
    rs=[]
    for f in files:
        d=dist(f)
        r=rssis(f,**kwargs)
        for v in r:
            ds.append(d)
            rs.append(v)
    return [np.array(ds),np.array(rs)]
def conf_mat(data,cv):
    rp=data[0]<=6*12
    pp=data[1]>=cv
    return [[np.sum(rp&pp),np.sum(rp&(~pp))],
         [np.sum((~rp)&pp),np.sum((~rp)&(~pp))]]
def prates(data,cv):
    #return (TPR, FPR)
    m=conf_mat(data,cv)
    return (m[0][0]/(m[0][0]+m[0][1]),m[1][0]/(m[1][0]+m[1][1]))
def plt_roc(ax,fs,R,**kwargs):
    D=data(fs,**kwargs)
    tprs=[]
    fprs=[]
    for cv in R:
        p=prates(D,cv)
        tprs.append(p[0])
        fprs.append(p[1])
    kwargs.pop('adver',None)
    ax.plot(fprs,tprs,**kwargs)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
def avg_box(ax,ei,**kwargs):
    boxes(ax,FSS[ei],**kwargs)
    avgs(ax,FSS[ei],**kwargs)
#     ax.set_xticks()
#     ax.set_xticklabels(x_ticks, rotation=0, fontsize=8)
    ax.set_xlabel('distance between advertiser and scanner (in)')
    ax.set_ylabel('RSSI')
    ax.set_title('averages and boxplots of experiment #'+str(str(ei)+'@'+str(kwargs['adver']) if 'adver' in kwargs else ei))
    
fig,ax=plt.subplots()
R=range(-90,-20)
cmap=plt.cm.get_cmap('tab20')
for i in range(1,len(FSS)):
    fs=FSS[i]
    f=fs[0]
    if f[f.find('@')+1:]=='1,2':
        for a in range(1,3):
            plt_roc(ax,fs,R,**{'alpha':0.5,'label':str(expi(f))+'@'+str(a),'adver':a,'color':cmap((i-1+0.5*(a-1))/len(FSS))})
    else:
        plt_roc(ax,fs,R,**{'alpha':0.5,'label':expi(f),'color':cmap((i-1)/len(FSS))})
ax.plot(range(2),'--',color='grey')
ax.legend()
ax.set_title('ROC curves')
plt.show()
#https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib
plt.savefig(DA_+'ROCs.png',bbox_inches='tight')
