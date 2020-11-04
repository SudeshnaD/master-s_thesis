

def listoflist(s):
    s=s.lstrip('[[')
    s=s.rstrip(']]')
    s=s.split('], [')
    xs=[x.split(', ') for x in s]
    def str_flt(l):
        l=[float(x) for x in l]
        return l
    lol=[]
    for x in xs:
        fltx=str_flt(x)
        lol.append(fltx)
    #lol.append([str_flt(x) for x in xs])
    return lol


with open('cos_sim_g.txt','r') as s:
    doc=s.readlines()
    doclist=listoflist(doc[0])
    print(len(doclist))

