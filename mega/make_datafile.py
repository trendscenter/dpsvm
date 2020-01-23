import random, scipy
import os
import numpy as np
from scipy import where, double, r_, c_, array, sign, dot, mean, randn
from scipy.io import loadmat
from sklearn.cross_validation import train_test_split
import tempfile
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from pylab import title, xlabel, ylabel, show, figure, ylim, demean, subplot, subplots_adjust, axis
import matplotlib.gridspec as gridspec

sites=10
rng = random.SystemRandom()
seed = rng.seed()
dpsvm = '../dpsvm'
dplogit = '../dplogit'
outfile = '/tmp/mega_svm.txt'
datadir = '/export/research/analysis/human/splis/confusion/vbm/fmri_data/'
BASE = '/export/research/analysis/human/splis/confusion/vbm/models/dbn_ng/data/'

def save_input_file(fname, data, labels, lmbd=0.01,
                    epsilon=10.0, huber=0.5):
    d = data.shape[1]
    if huber < 0:
        params = [data.shape[0], d, lmbd, epsilon]
    else:
        params = [data.shape[0], d, lmbd, epsilon, huber]
    with open(fname, 'w') as f:
        for c in params:
            f.write(str(c)+"\n")
        for c in data:
            f.write(" ".join(map(str, c))+"\n")
        for c in labels:
            f.write(str(c)+"\n")

def save_svm_input(fname, data, labels, lmbd=0.01,
                    epsilon=10.0, huber=0.5):
    save_input_file(fname, data, labels, lmbd=lmbd,
                    epsilon=epsilon, huber=huber)

def save_logit_input(fname, data, labels, lmbd=0.01,
                     epsilon=10.0):
    save_input_file(fname, data, labels, lmbd=lmbd,
                    epsilon=epsilon, huber=-1)

def solvesvms(infile, solver=dpsvm):
    ws = {'svm':[],'out':[],'obj':[]}
    svms = ['obj','out','svm']
    f = open(infile)
    l = f.readlines()
    d = int(l[1])
    f.close()
    for w in os.popen(solver+' '+infile):
        if w[-2] == '0':
            ws[svms.pop()] = array(map(double,w.split(' ')[0:d]))
        else:
            k = svms.pop()
            ws[k] = array(map(double,w.split(' ')[0:d]))
            ws[k][:] = 1000.
            print 'convergence problem'
            #raise Exception("SOLVER FAILED TO CONVERGE!")
    return ws

def solvelogit(infile, solver=dplogit):
    ws = {'lr':[0],'out':[0],'obj':[0]}
    lrs = ['obj','out','lr']
    f = open(infile)
    l = f.readlines()
    d = int(l[1])
    f.close()

    for w in os.popen(solver+' '+infile):
        if w[-2] == 0:
            k = lrs.pop()
            ws[k] = array(map(double,w.split(' ')[0:d]))
        else:
            k = lrs.pop()
            ws[k] = array(map(double,w.split(' ')[0:d]))
            if sum(abs(ws[k])) == 0:
                os.remove(ofile)
                raise RuntimeError('Zero decision boundary')
                #ws[k] = randn(max(ws[k].shape))
            #ws[k][:] = 1000.
            #print 'convergence problem'
            #raise Exception("SOLVER FAILED TO CONVERGE!")
    return ws

def test_errors(ws, data, labels):
    error = {}
    for v in ws:
        error[v] = 100*abs(sum(map(lambda x: min(0,x), sign(dot(ws[v],data.T))
                                   * labels)))/labels.shape[0]
    return error

def dpsvmsolve(data, labels, test, tlabels, eps = 5, lmbd = 0.01):
    # compute SVM solution
    f = tempfile.NamedTemporaryFile(delete=True)
    f.close()
    ofile = f.name
    save_input_file(ofile, data, labels, epsilon = eps, lmbd = lmbd)
    w = solvesvms(ofile)
    e = test_errors(w, test, tlabels)
    os.remove(ofile)
    return (w,e)

def dplrsolve(data, labels, test, tlabels, eps = 5, lmbd = 0.01):
    # compute LR solution
    f = tempfile.NamedTemporaryFile(delete=True)
    f.close()
    ofile = f.name
    save_logit_input(ofile, data, labels, epsilon = eps, lmbd = lmbd)
    w = solvelogit(ofile)
    e = test_errors(w, test, tlabels)
    os.remove(ofile)
    return (w,e)

def halfsplit(data, labels, rnds=rng.seed()):
    d1, d2, l1, l2 = train_test_split(data, labels,
                                      test_size=0.5, random_state=rnds)
    return [(d1,l1),(d2,l2)]

def multisplit(data, labels, parts, rnds=rng.seed()):
    if parts != parts/2 * 2:
        d1, d2, l1, l2 = train_test_split(data, labels,
                                          test_size=1.0/parts,
                                          random_state=rnds)
        return multisplit(d1, l1, parts-1, rnds=rnds) +\
               [(d2,l2)]
    if parts == 2:
        return halfsplit(data, labels, rnds = rnds)
    else:
        p1, p2 = halfsplit(data, labels, rnds = rnds)
        return multisplit(p1[0], p1[1], parts/2, rnds=rnds) +\
               multisplit(p2[0], p2[1], parts/2, rnds=rnds)

def multisplit_public(data, labels, parts, pub=0.3, rnds=rng.seed()):
    d1, d2, l1, l2 = train_test_split(data, labels,
                                      test_size=pub,
                                      random_state=rnds)
    return multisplit(d1, l1, parts-1, rnds=rnds) +\
           [(d2,l2)]

def train_test_multisplit(data, labels, parts, ts=0.1, rnds = rng.seed()):
    d, td, l, tl = train_test_split(data, labels,
                                    test_size=ts,
                                    random_state=rnds)
    return {'train': multisplit(d, l, parts), 'test': (td,tl)}

def tt_pub_multisplit(data, labels, parts, pub=0.3, ts=0.1, rnds = rng.seed()):
    if ts == 0.0:
        return {'train': multisplit_public(data, labels, parts, pub=pub)}
    d, td, l, tl = train_test_split(data, labels,
                                    test_size=ts,
                                    random_state=rnds)
    return {'train': multisplit_public(d, l, parts, pub=pub),
            'test': (td,tl)}

def listr(l):
    return map(lambda *a: list(a), *l)

def data2data(data, clfs, kind='svm'):
    """
    Applies all classifiers to the data and returns their output as vectors
    """
    W=array([clfs[i][0][kind] for i in range(0,len(clfs))])
    X = dot(data, W.T)
    X = demean(X,axis=1)
    return X

def stacked_classifier(splt, eps=15):
    test = splt['test']
    sols = []
    for r in splt['train']:
        # compute SVM solution at a site
        sols = sols + [dpsvmsolve(r[0], r[1], test[0], test[1], eps=eps)]

    e = {}
    for kind in sols[0][0]:
        # train
        data = data2data(splt['train'][-1][0], sols[:-1], kind = kind)
        clf = LogisticRegression()
        clf.fit(data, splt['train'][-1][1])#splt['train'][-1][1])
        # test
        data = data2data(test[0], sols[:-1], kind = kind)
        e[kind] = 100*abs(sum(map(lambda x: min(0,x),
                                  clf.predict(data)*test[1])))/double(len(test[1]))
#        e[kind] = 100*len(where(clf.predict(data)*test[1]==-1)[0])/double(len(test[1]))

    return ([v[1]['obj'] for v in sols]+[e['obj']],
            [v[1]['svm'] for v in sols]+[e['svm']],
            [v[1]['out'] for v in sols]+[e['out']])

def p_stacked_classifier(splt, eps=15):
    test = splt['test']
    sols = []
    for r in splt['train']:
        sols = sols + [dpsvmsolve(r[0], r[1], test[0], test[1], eps=eps)]
    e = {}
    for kind in sols[0][0]:
        # train
        data1 = data2data(splt['train'][-1][0], sols[:-1], kind = kind)
        # test
        data2 = data2data(test[0], sols[:-1], kind = kind)
        completed = False
        lll = 0.5
        stp = 0.001
        while not completed:
            try:
                clf = dplrsolve(data1, splt['train'][-1][1],
                                data2, test[1], eps=eps, lmbd = lll)
                completed = True
            except Exception:
                lll -= stp
                pass
        if kind == 'svm':
            e['lr'] = clf[1]['lr']
        else:
            e[kind] = clf[1][kind]
    return ([v[1]['obj'] for v in sols]+[e['obj']],
            [v[1]['svm'] for v in sols]+[e['lr']],
            [v[1]['out'] for v in sols]+[e['out']])

g1 = np.load(BASE + 'rbm3cdd_LAST/train/'+'hidden1-00001-of-00001.npy')
g2 = np.load(BASE + 'rbm3cdd_LAST/train/'+'hidden2-00001-of-00001.npy')
g3 = np.load(BASE + 'rbm3cdd_LAST/train/'+'hidden3-00001-of-00001.npy')

gt1 = np.load(BASE + 'rbm3cdd_LAST/test/'+'hidden1-00001-of-00001.npy')
gt2 = np.load(BASE + 'rbm3cdd_LAST/test/'+'hidden2-00001-of-00001.npy')
gt3 = np.load(BASE + 'rbm3cdd_LAST/test/'+'hidden3-00001-of-00001.npy')

y = np.load(datadir + 'vbm_train_labels_3mm_voxnorm.npy')
yt = np.load(datadir + 'vbm_test_labels_3mm_voxnorm.npy')

X = r_[g1,gt1]
y = r_[y,yt]
y[where(y==0)]=-1

X = X - mean(X,axis=0)
obj=[]
svm=[]
out=[]

# average classifier
# for i in range(0,100):
#     splt = train_test_multisplit(X, y, sites, ts=0.3, rnds=rng.seed())
#     test = splt['test']
#     sols = []
#     for r in splt['train']:
#         # compute SVM solution at a site
#         sols = sols + [dpsvmsolve(r[0], r[1], test[0], test[1], eps=15)]

#     ww={}
#     for v in sols[0][0]:
#         ww[v] = scipy.sum([i[0][v] for i in sols],axis=0)
#         e = test_errors(ww, test[0], test[1])
#     obj.append([v[1]['obj'] for v in sols]+[e['obj']])
#     svm.append([v[1]['svm'] for v in sols]+[e['svm']])
#     out.append([v[1]['out'] for v in sols]+[e['out']])

#X, tX, y, ty = train_test_split(X, y, test_size=0.3, random_state=rng.seed())
#tst = (tX,ty)
for i in range(0,100):
    splt = tt_pub_multisplit(X, y, sites+1, ts=0.3, pub=1./(sites+1), rnds=rng.seed())
    #splt = {'train':multisplit(X, y, sites+1, rnds=rng.seed())}
    #splt = train_test_multisplit(X, y, sites+1, ts=0.3, rnds=rng.seed())
    #splt['test'] = tst
    e = stacked_classifier(splt, eps=10)
    obj.append(e[0])
    svm.append(e[1])
    out.append(e[2])

obj = listr(obj)
svm = listr(svm)
out = listr(out)

#pal = sns.hls_palette(sites+2, s=.75)
pal = sns.color_palette("Set2_r", sites+2)

f = figure(figsize=[9,3.5])
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1]) 
ax1 = subplot(gs[0])

sns.boxplot(obj,names=[str(i) for i in range(1,sites+1)]+['agg']+['combo'], widths=.5,ax=ax1, color=pal)
#title('private data private aggregator error rate distributions')
#xlabel('sites with private data public data aggregator combined classifier probability')
ylabel('classification error rate %')
ax2 = subplot(gs[1], sharey=ax1)
sns.distplot(obj[-1],hist=True,kde_kws={"lw": 1, "color": pal[-1]},legend=False, vertical=True,ax=ax2,color=pal[-1])
for i in range(0,sites+1):
    sns.distplot(obj[i],hist=False,kde_kws={"lw":1},legend=False, vertical=True,ax=ax2, color=pal[i])
xlabel('probability')
#xlim([0,0.35])
ylim([-5,60])
axis("off")
subplots_adjust(left=0.08,right=0.98, wspace=0.01)
show()
