import math
import numpy
from itertools import product, combinations
import cv2 as cv
from math import floor
from numpy import ndarray
import matplotlib.pyplot as plt
# paths
resource = 'Resource\\'
result = 'Result\\'
light_diections = resource + 'light_directions.txt'
normal_reel= resource + 'normal.txt'
light_intensities = resource + 'light_intensities.txt'
mask = resource + 'mask.png'
filenames = resource + 'filenames.txt'
matrix_e = result + 'mat_e.DONOTOPEN'
normales = result + 'normales.txt'


def load_light_sources():
    global light_diections
    file = open(light_diections, 'r')#ouvrir fichier lightdirection.txt 
    arr = []
    while True:
        line = file.readline()#lire le fichier ligne par ligne 
        if not line:
            break
        temp = line.strip().split(' ')#decouper la ligne par rapport a le vide

        temp = [float(x) for x in temp] # Change from str to float

        arr.append(temp)#on a jouter l element dans la list 
    file.close()
    return arr


def load_intense_sources():
    global light_intensities
    file = open(light_intensities, 'r')#ouvrir  le fichier  light intensities.txt
    arr = []

    while True:
        line = file.readline()#lire le fichier ligne par ligne 
        if not line:
            break
        temp = line.strip().split(' ')
        temp = [float(x) for x in temp]  # replace str with float
        arr.append(temp)
    file.close()
    return arr


def load_obj_mask():
    global mask
    mat = cv.imread(mask, cv.IMREAD_GRAYSCALE) #import le mask en grayscale
    rows, columns = mat.shape #recuperer les dimension de mask
    ceil = 255 // 2     # default ceil

    for i in range(columns):
        for j in range(rows):
            if mat[j, i] < ceil:
                mat[j, i] = 0 #si pixel courant < ceil alors en le remplace par 0
            else:
                mat[j, i] = 1 #sinon 1 
    return mat


def treat_image(img_file, intense_row):
    # Read
    image = cv.imread(img_file, cv.IMREAD_UNCHANGED)
    # Normalize float 32 from 0 to 1
    image = cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    rows, columns, _ = image.shape
    # Define gray image shapes
    image_gray = numpy.zeros((rows, columns), dtype=numpy.float32)
    for i in range(columns):
        for j in range(rows):
            # image(BGR) is divide by intensity (RGB)
            image[j, i] = image[j, i, 0] / intense_row[2], image[j, i, 1] / intense_row[1], image[j, i, 2] /intense_row[0]
            # Change to grey image
            image_gray[j, i] = ((image[j, i, 0]*.3) + (image[j, i, 1]*.59) + (image[j, i, 2]*.11)) / 3
    
    # Reshape into 1 row and return
    return image_gray.reshape(1, rows * columns)[0]
    

def create_matrix_e():  # save matrix E
    global matrix_e, filenames, resource
    file = open(matrix_e, "w")
    # Read intensity file into list
    intense = load_intense_sources()
    # Read pictures filenames
    filenames = open(filenames)
    # output list: contains list of (N, w*h) of one lined images
    images_list = list()
    itr = 0
    while True:
        filename = filenames.readline().strip()
        if not filename:
            # end of file (filenames.txt)
            break
        treat = treat_image(resource + filename, intense[itr])
        itr += 1
        print(itr)
        for elem in treat:
            file.write(str(elem) + " ")
        file.write("\n")
        images_list.append(treat)
    file.close()
    filenames.close()
    return images_list

def load_matrix_e():
    global matrix_e
    file = open(matrix_e, 'r')  #charger le fichier de la matrice E
    arr = []
    while True:
        line = file.readline()   #lire le fichier ligne par ligne
        if not line:
            break
        temp = line.strip().split(' ')  #decouper la ligne par rapport le vide
        temp = [float(x) for x in temp] # Change from str to float
        arr.append(temp)
    file.close()
    return arr


def load_images(flag):
    if flag == 1:
        return create_matrix_e()  # creates the file (E matrix)
    else:
        return load_matrix_e()  # loades the file if already created


def calcul_needle_map():    # creates n wich is S^-1 * E
    global matrix_e, normales
    #obj_images = load_images(2)
    light_sources = load_light_sources()
    #obj_masques = load_obj_mask()
    file = open(normales, "w")
    file2 = open(matrix_e, "r")

    matE = list()
    # create inverted S
    s_inv = numpy.linalg.pinv(light_sources)

    # read matrix E
    while True:
        line = file2.readline()
        if not line:
            break
        arr = line.strip().split(' ')
        arr = [float(x) for x in arr]
        matE.append(arr)

    # multiply s^-1 * E
    n=numpy.matmul(s_inv,matE)
    sum=n[0,:]*n[0,:]#calcule modulo ||N||
    for i in range(1,3):
        sum+=n[i,:]*n[i,:]
    s1=numpy.sqrt(sum)
    for i in range(3):
        n[i,:] = n[i,:] / s1    


    # write down n
    for i in range(3):
        for j in range(612*512):
           file.write(str(n[i,j]) + " ")
        file.write("\n")
    file.close()
    file2.close()
    #cv.normalize(n,n,0,255,cv.NORM_MINMAX,cv.CV_8U)
   
    
    return n


def load_normales():
    global normales
    file = open(normales , 'r') # ouvrire le fichier qui contien la matrice Normal
    arr = []
    while True:
        line = file.readline()
        if not line:
            break
        temp = line.strip().split(' ')
        temp = [float(x) for x in temp] # Change from str to float
        arr.append(temp)
    return arr

def normales_to_img(n=None):
    mask=load_obj_mask()#charger le mask
    img_result= numpy.zeros((512, 612,3),numpy.uint8)
    if n is None:
        n=load_normales() #charger la matrice normal
    n =numpy.array(n)
    k = 0
    
    for i in range(512):
        for j in range(612):
                #normalisation
             img_result[i,j,0]=((n[2,k]+1)/2)*255*mask[i,j]
             img_result[i,j,1]=((n[1,k]+1)/2)*255*mask[i,j]
             img_result[i,j,2]=((n[0,k]+1)/2)*255*mask[i,j]
             k+=1
    
    
    cv.imshow("resultat ", img_result) #afficher image resultat apres normalisation
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return img_result


def calcule_z_a_partir_de_vecteur_normal(n):
    res_p=list()
    res_q=list()
    q=0
    p=0
    mask=load_obj_mask() # import le mask 
   
    for k in range(512*612):
            p=-n[0,k]/n[2,k] #calcule de p=-x/z
            q=-n[1,k]/n[2,k]    #calcule de  q=-y/z
            res_q.append(q)
            res_p.append(p)
    
    znx=numpy.zeros((512, 612),numpy.float32)   #matrice pour l axe x
    zny=numpy.zeros((512, 612),numpy.float32) #matrice pour l axe y
    res_p=numpy.reshape(res_p,(512,612))
    res_q=numpy.reshape(res_q,(512,612))
    res_p=mask*res_p #multuplie mask fois la list de p et q
    res_q=mask*res_q
    for i in range(1,512):
            znx[i,:]=znx[i-1,:]+res_p[i,:]#zn =z(n-1) + p(courant selon le vecteur n) pour axe x
    for j in range(1,612):
            zny[:,j]=znx[:,j-1]+res_q[:,j] #zn =z(n-1) + q(courant selon le vecteur n) pour axe y
    
    plt.figure()
    x=numpy.array(znx)
    y=numpy.array(zny)
    x=x.reshape((512,612))
    y=y.reshape((512,612))
    ax = plt.axes(projection='3d') 
    x1=range(612) # initialiser les valeurs de x1 (axe X)
    y1=range (512)# initialiser les valeurs de y1 (axe Y)
    x1,y1=numpy.meshgrid(x1,y1) #affecter les valeurs au plan x ,y
    m=mask*(x+y)/2# pour axe Z on a pris la moyenne des deux autres X et Y
    ax.plot_surface(x1,y1,m) # pour affecter les valeurs pour les axes 
    plt.show()
   
    #normalisation max min 
    max_ = m.max()
    min_ = m.min()
    m = 255*(m - min_) / (max_ - min_)   
            
    cv.imshow("image composant Z",m.astype("uint8"))# afficher l image de Z en uint8
    cv.waitKey(0)
    return         

def load_normal_reel():
    file = open(normal_reel, 'r')#ouvrir le fihcier normal.txt valeur reel
    arr = numpy.zeros((3,512*612),dtype=numpy.float32)
   
    for i in range(612*512):
            line1 = file.readline() #lire la premier valeur x
            
            line2 = file.readline() #lire 2 eme valeur y
            
            line3 = file.readline() #lire la 3 eme  valeur z
            arr[:,i]=float(line1),float(line2),float(line3)
    file.close()
    return arr

def bonus_calcule_erreur ():
    n_est=calcul_needle_map()
    n_true =load_normal_reel()
    dot_m = numpy.zeros((512*612),dtype=numpy.float32)
    
    for i in range(612*512):
        dot_m[i]=numpy.dot(n_true[:,i],n_est[:,i])
        dot_m[i]=180*math.acos(dot_m[i])/math.pi
    img_result= numpy.zeros((512, 612),numpy.uint8)
    k = 0    
    for i in range(512):
        for j in range(612):
                #normalisation
             img_result[i,j]=((dot_m[k]+1)/2)*255
             k+=1
    
    
    cv.imshow("fenetre resultat erreur", img_result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
   


   # normales_to_img(dot_m)  
      
    return 


    
    

if __name__ == "__main__":
    
    
    while True :
        print("Que voulez-vous faire ?")
        print(" 1 - Charger les images. \n 2 - Calculer les normales de l'objet. \n 3 - Afficher les normales dans une image. \n 4 - Calculer la profondeur Z et afficher l'objet 3D. \n 5 - afficher image d erreur.\n0 - Quitter.")
        answer = input("")
        if answer == '0':
            break
        if answer == '1':
            answer = input("   1 - fichier pas encore crée, je veux le créer. \n   2 - fichier crée , je veux le charger.\n  0 - Retour.\n")
            if answer != '0':
                load_images(int(answer))
        if answer == '2':
            print(" Calcule des normales en cours...\n")
            calcul_needle_map()
        if answer == '3':
            print(" Affichage des normales dans une image...\n")
            normales_to_img()
        if answer == '4':
            print(" Calcule de la profondeur Z... \n Affichage de l'objet 3D...\n")
            calcule_z_a_partir_de_vecteur_normal(calcul_needle_map())
        if answer == '5':
            print(" Calcule de la matrice  d erreur ... \n Affichage de l image d erreur ...\n")
            bonus_calcule_erreur()    