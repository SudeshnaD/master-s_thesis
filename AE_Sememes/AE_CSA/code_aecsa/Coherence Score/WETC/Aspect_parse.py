def aspect_parse(asp_size):

    aspect_file='/home/sdasgupta/sdasgupta_dir/Sememes_AE-SA/experiment_sentihood/GloveCode/code_aecsa/output_dir/aecsa/aspect.log'+str(asp_size)
    file=open(aspect_file)

    asp=[line.split(' ') for line in file if 'Aspect' in line]
    asp=[x[1] for x in asp]
    asp=[x.split(':') for x in asp]
    asp=[int(x[0]) for x in asp]
    aspect_dict={x:'' for x in asp}


    with open(aspect_file,'r') as f:
        reader=f.read()

    for i,part in enumerate(reader.split("Aspect")):
        if i==0:
            pass
        else:
            aspect_dict[i-1]=part


    x=lambda x : x.lstrip(' 0:\\')


    for i,v in enumerate(aspect_dict.values()):
        s=list(aspect_dict.values())[i]
        #s=x(s)
        s=s.rstrip('\n\n')
        s=s.split(':\n')[1]
        aspect_dict[i]=s


    for i,v in enumerate(aspect_dict.values()):
        s=list(aspect_dict.values())[i]
        #s=s.lstrip('\n')
        s=s.split(' ')
        aspect_dict[i]=s
    

    for i,v in enumerate(aspect_dict.values()):
        s=list(aspect_dict.values())[i]
        s=[j for i in [x.split('|') for x in s] for j in i]
        s=s[::2]
        aspect_dict[i]=s


    #print(aspect_dict)

    import pickle
    with open('aspect_file'+str(asp_size)+'.pkl','wb') as f:
        pickle.dump(aspect_dict, f)

    return aspect_dict