

#prova= [[], ['F#-2', 'F#-3'], ['E-2', 'F#-3', 'G#-2'], ['G#-2', 'E-2'], ['G#-2', 'E-2'], ['E-1', 'F-1', 'G#-1'], ['F-1'], ['F#-1', 'G#-1'], ['G#-1', 'F#-1'],['F#-2'],[],[]]
prova=[['F#-2'],['F#-2'],['F#-2'],['F#-2'],['F#-2'],['F#-2'],['F#-2'],['F#-2'],['F#-2'],['F#-2'],['F#-2'],['F#-2'],['F#-2']]

def good_time(n):
        lst=[0,1,2,4,8,16,32]
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-n))]
        
#just add intervals to the note
def magic_spartito(perception):    # note con intervalli più lunghi di 8 da fixare
    

    dizio={"A-0":0, "A#-0":0, "B-0":0, "C-1":0, "C#-1":0, "D-1":0, "D#-1":0, "E-1":0, "F-1":0, "F#-1":0, "G-1":0, "G#-1":0, "A-1":0, "A#-1":0, "B-1":0,
            "C-2":0, "C#-2":0, "D-2":0, "D#-2":0, "E-2":0, "F-2":0, "F#-2":0, "G-2":0, "G#-2":0, "A-2":0, "A#-2":0, "B-2":0,
            "C-3":0, "C#-3":0, "D-3":0, "D#-3":0, "E-3":0, "F-3":0, "F#-3":0, "G-3":0, "G#-3":0, "A-3":0, "A#-3":0, "B-3":0,
            "C-4":0, "C#-4":0, "D-4":0, "D#-4":0, "E-4":0, "F-4":0, "F#-4":0, "G-4":0, "G#-4":0, "A-4":0, "A#-4":0, "B-4":0,
            "C-5":0, "C#-5":0, "D-5":0, "D#-5":0, "E-5":0, "F-5":0, "F#-5":0, "G-5":0, "G#-5":0, "A-5":0, "A#-5":0, "B-5":0,
            "C-6":0, "C#-6":0, "D-6":0, "D#-6":0, "E-6":0, "F-6":0, "F#-6":0, "G-6":0, "G#-6":0, "A-6":0, "A#-6":0, "B-6":0,
            "C-7":0, "C#-7":0, "D-7":0, "D#-7":0, "E-7":0, "F-7":0, "F#-7":0, "G-7":0, "G#-7":0, "A-7":0, "A#-7":0, "B-7":0, "C-8":0}

    result = [[] for i in range(len(perception))]
    repeated=set()
    for i,note_list in enumerate(perception):
        if note_list or repeated:
            for note in repeated.copy():
                if note not in note_list or dizio[note]==8:
                    result[i-dizio[note]].append((note,good_time(dizio[note]),0))  #note interval and additional time for the legature
                    dizio[note]=0
                    repeated.remove(note)
            for note in note_list:
                if note not in repeated:
                    repeated.add(note)
                    dizio[note]+=1
                else:
                    dizio[note]+=1
        else:
            result[i]=[]
    return result


def divide_spartito(magic_spartito):   #da testare
    cleff_violin=[[] for i in range(len(magic_spartito))]
    cleff_bass=[[] for i in range(len(magic_spartito))]
    for i,note_list in enumerate(magic_spartito):
        for note in note_list:
            if int(note[0][-1])==4:
                if int(note[0][1]<='c'):
                    cleff_bass[i].append(note)
                else:
                    cleff_violin[i].append(note)
            elif int(note[0][-1])>4:
                cleff_violin[i].append(note)
            else:
                cleff_bass[i].append(note)
    return cleff_violin,cleff_bass

                



def add_legature(magic_spartito):
    #order in ascending order the notes in respect to the note interval
    for c,element in enumerate(magic_spartito.copy()):
        if element:
            magic_spartito[c].reverse()
    c=0
    for t,note_list in enumerate(magic_spartito.copy()):
        if c==0 and note_list:
            times=[element[1] for element in note_list]
            c=times[0]-1
            if set([good_time(time) for time in times])!=1:
                new=[]
                shortest_interval=times[-1]
                magic_spartito[t] = [(note[0],shortest_interval,good_time(note[1]-shortest_interval)) for note in note_list]
            

        else:
            if c>0: c-=1 
            magic_spartito[t] = []
            
    return magic_spartito       
                

def to_lilypond(magic_spartito):
    def time_to_lilipond(n):
        return int(8/n)

    result=""
    c=0
    pause_counter=0
    for note_list in magic_spartito:
        if note_list:
            if pause_counter>0:
                
                if pause_counter == 6:
                    result+=" r2 r4"
                else:
                    result+=f" r{time_to_lilipond(good_time(pause_counter))}"
                pause_counter=0
            c=note_list[0][1]+note_list[0][2]-1
            result+=" <<"
            for note in note_list:
                result=result+" {"+note[0][0].lower()
                if note[0][1] == "#":
                    result+="is"

                result+=int(note[0][-1])*"'"+str(time_to_lilipond(note[1]))
                if note[2]!=0:
                    result+=f"~{time_to_lilipond(note[2])}"
                result+="}"
            result+=">>"
        else:
            if c==0:
                pause_counter+=1
                if pause_counter==8:
                    pause_counter=0
                    result+=" r1"
            else:
                c-=1
        #reuse of the if written before for the last case
    if pause_counter>0:
        if pause_counter == 6:
            result+=" r2 r4"
        else:
            result+=f" r{time_to_lilipond(good_time(pause_counter))}"
    return result



# prova=magic_spartito(prova)
# print(prova)
# spartito=add_legature(prova)
# print(prova)
# spartito=to_lilypond(prova)      
# print(prova)
def rotellini(prova,path=""):
    spartito=magic_spartito(prova)
    spartito1,spartito2 = divide_spartito(spartito)
    spartito1=to_lilypond(add_legature(spartito1))
    spartito2=to_lilypond(add_legature(spartito2))


    text=r"upper = \fixed a,,, "+" { \n\clef treble\n\key c \major\n"+ r'\time 4/4'+"\n\n"
    text+=spartito1+"\n\n"
    text+="}\n"+r"lower = \fixed a,,, "+" {\n\clef bass\n\key c \major\n"+r"\time 4/4"+"\n\n"
    text+=spartito2+"\n\n"
    text+='}\n\score {\n'+r'\new'+' PianoStaff \with { instrumentName = "Piano" }\n<<\n'+r'\new' + r' Staff = "upper" \upper'+'\n' + r"\new" + ' Staff = "lower" \lower\n>>\n\layout { }\n\midi { }\n}'

    f = open(f"{path}spartito.ly", "w")
    f.write(text)
    f.close()

    print(text)

    import os
    os.startfile(f"{path}spartito.ly")
    return 


"""
\version "2.22.2"
\fixed a,,, {
  \clef violin
  \time 4/4

 r8 << {fis'''8~8} {fis''8}>> r4 << {f'8~8} {e'8} {gis'8}>> << {gis'4} {fis'4}>> r4

% a b b' a' a''''
}


"""