from IPython.core.display import display, HTML,display_html
import ipywidgets as widgets
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from ipywidgets import interact, interact_manual
from IPython.display import display, Markdown, clear_output

def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

def show_simple_models_thresholds(df,coluna_eixo_y,ratings_eixo_y,coluna_eixo_x,ratings_eixo_x,VAR_BAD):
    if ratings_eixo_y==None:
        try:
            ratings_eixo_y = sorted(list(df[coluna_eixo_y].unique()))
        except:
            ratings_eixo_y = list(df[coluna_eixo_y].unique())
    
    if ratings_eixo_x==None:
        try:
            ratings_eixo_x = sorted(list(df[coluna_eixo_x].unique()))
        except:
            ratings_eixo_x = list(df[coluna_eixo_x].unique())
    
    final = pd.concat([\
    df[[coluna_eixo_y,VAR_BAD]].groupby([coluna_eixo_y],as_index=False).sum(),
    (df[[coluna_eixo_y,VAR_BAD]].groupby([coluna_eixo_y],as_index=False).count()).rename(index=str,columns={VAR_BAD:'hum'})['hum'].reset_index(drop=True)],axis=1)
    final[VAR_BAD+'perc'] = (final[VAR_BAD]/final['hum'])
    #
    indices = ratings_eixo_y
    #
    final = final.set_index(coluna_eixo_y).loc[indices]
    #
    data = []
    for j in np.arange(1,final.shape[0]+1):
        bad_total = np.sum(final.values[:j,0])/np.sum(final.values[:j,1])
        sum_pop = np.sum(final.values[:j,1])/np.sum(final.values[:,1])
        limiar = indices[j-1]
        data.append([limiar, bad_total, sum_pop])
    resultados = pd.DataFrame(data,columns=['LimiarMax','TARGET(%)','POP(%)'])

    final2 = pd.concat([\
    df[[coluna_eixo_x,VAR_BAD]].groupby([coluna_eixo_x],as_index=False).sum(),
    (df[[coluna_eixo_x,VAR_BAD]].groupby([coluna_eixo_x],as_index=False).count()).rename(index=str,columns={VAR_BAD:'hum'})['hum'].reset_index(drop=True)],axis=1)
    final[VAR_BAD+'perc'] = (final[VAR_BAD]/final['hum'])
    #
    indices = ratings_eixo_x
    #
    final2 = final2.set_index(coluna_eixo_x).loc[indices]
    #
    data = []
    for j in np.arange(1,final2.shape[0]+1):
        bad_total = np.sum(final2.values[:j,0])/np.sum(final2.values[:j,1])
        sum_pop = np.sum(final2.values[:j,1])/np.sum(final2.values[:,1])
        limiar = indices[j-1]
        data.append([limiar, bad_total, sum_pop])
    resultados2 = pd.DataFrame(data,columns=['MaxThresh','TARGET(%)','POP(%)'])

    display_side_by_side(resultados,resultados2)
    
def create_support_matrices(df,base_bad,colmodelo1,colmodelo2,indexmodel1=None,indexmodel2=None):
    if indexmodel1==None:
        try:
            indexmodel1 = sorted(list(df[colmodelo1].unique()))
        except:
            indexmodel1 = list(df[colmodelo1].unique())
    
    if indexmodel2==None:
        try:
            indexmodel2 = sorted(list(df[colmodelo2].unique()))
        except:
            indexmodel2 = list(df[colmodelo2].unique())
    ###########################################################################################################
    aglomeracaobad = df[[colmodelo1,colmodelo2,base_bad]].groupby([colmodelo1,colmodelo2],as_index=False).sum()
    matriz_migracaobad = pd.DataFrame([],index=indexmodel1,columns=indexmodel2).fillna(0)
    for x in aglomeracaobad.iterrows():
        matriz_migracaobad[x[1][1]].loc[x[1][0]] = x[1][2]
    ###########################################################################################################
    aglomeracaobad = df[[colmodelo1,colmodelo2,base_bad]].groupby([colmodelo1,colmodelo2],as_index=False).count()
    matriz_migracaobad2 = pd.DataFrame([],index=indexmodel1,\
                                      columns=indexmodel2).fillna(0)
    for x in aglomeracaobad.iterrows():
        matriz_migracaobad2[x[1][1]].loc[x[1][0]] = x[1][2]
    
    return ((matriz_migracaobad/matriz_migracaobad2)).fillna(0),matriz_migracaobad, matriz_migracaobad2

def generate_thresholds(matrizbadfinal, matrizpop, Bad_Rate):
    
    indices = list(matrizbadfinal.index)
    colunas = list(matrizbadfinal.columns)
    data = []
    for i in np.arange(1,matrizbadfinal.shape[0]+1):
        for j in np.arange(1,matrizbadfinal.shape[1]+1):
            bad_total = int(np.sum(matrizbadfinal.values[:i,:j]*matrizpop.values[:i,:j]))/np.sum(matrizpop.values[:i,:j])
            sum_pop = np.sum(matrizpop.values[:i,:j]/np.sum(matrizpop.values))
            limiar = str(indices[i-1])+' '+str(colunas[j-1])
            data.append([limiar, bad_total, sum_pop,i,j])
            
    resultados = pd.DataFrame(data,columns=['Suggested threshold','TARGET(%)','POP(%)','index','columns'])
    display(resultados[['Suggested threshold','TARGET(%)','POP(%)']][(resultados['TARGET(%)'] < Bad_Rate/100)].sort_values(by='TARGET(%)',ascending=False).head())
    #Ploting
    plt.figure(figsize=(10,6))
    plt.title('TARGET X Population')
    plt.scatter(resultados['TARGET(%)']*100,resultados['POP(%)']*100)
    plt.xlabel('TARGET (%)')
    plt.ylabel('Population (%)')
    plt.show()
    return resultados[(resultados['TARGET(%)'] < Bad_Rate/100)].sort_values(by='TARGET(%)',ascending=False).reset_index(drop=True)
    
def create_grid(matrizbadfinal,resultados):
    grid = widgets.GridspecLayout(matrizbadfinal.shape[0]+1, matrizbadfinal.shape[1]+1,max_width=5)
    for i in range(matrizbadfinal.shape[1]):
        grid[0,i+1] = widgets.Text(value=str(matrizbadfinal.columns[i]), disabled=True,layout=widgets.Layout(width='100px'))
    for j in range(matrizbadfinal.shape[0]):
        grid[j+1,0] = widgets.Text(value= str(matrizbadfinal.index[j]), disabled=True, layout=widgets.Layout(width='100px'))

    for i in range(matrizbadfinal.shape[0]):
        for j in range(matrizbadfinal.shape[1]):
            if i < resultados['index'].values[0] and j < resultados['columns'].values[0]:
                grid[i+1,j+1] = widgets.ToggleButton(description=str(np.round(matrizbadfinal.values[i,j]*100,1))+"%",value=True,layout=widgets.Layout(width='100px'))
            else:
                grid[i+1,j+1] = widgets.ToggleButton(description=str(np.round(matrizbadfinal.values[i,j]*100,1))+"%",value=False,layout=widgets.Layout(width='100px'))
    return grid

def show_pol(grid,matrizbadfinal,matrizbadpop,matrizpop,matrizbadswap0,matrizpopswap0,matrizbadswap1,matrizpopswap1,var_swap,my_cmap):
    #Calculando BadRate no ponto, Swapin, Swapout, etc...
    mx_pol = np.zeros((matrizbadfinal.shape))
    mx_pol_out = np.zeros((matrizbadfinal.shape))

    for i in range(matrizbadfinal.shape[0]):
        for j in range(matrizbadfinal.shape[1]):
            mx_pol[i,j] = grid[i+1,j+1].value
            mx_pol_out[i,j] = not grid[i+1,j+1].value

    #
    BAD_IN = np.sum(np.sum((matrizbadpop.values*mx_pol)))/np.sum(np.sum((matrizpop.values*mx_pol)))
    BAD_OUT = np.sum(np.sum((matrizbadpop.values*mx_pol_out)))/np.sum(np.sum((matrizpop.values*mx_pol_out)))
    
    POP_IN = np.sum((matrizpop.values*mx_pol))/np.sum(matrizpop.values)
    POP_OUT = np.sum((matrizpop.values*mx_pol_out))/np.sum(matrizpop.values)
    #calculando swapin
    if var_swap != None:
        SWAPIN = np.sum((matrizbadswap0.values*mx_pol))/np.sum((matrizpopswap0.values*mx_pol))
        SWAPOUT = np.sum((matrizbadswap1.values*mx_pol_out))/np.sum((matrizpopswap1.values*mx_pol_out))
        ININ = np.sum((matrizbadswap1.values*mx_pol))/np.sum((matrizpopswap1.values*mx_pol))
        OUTOUT = np.sum((matrizbadswap0.values*mx_pol_out))/np.sum((matrizpopswap0.values*mx_pol_out))
    
        POPSWAPIN = np.sum((matrizpopswap0.values*mx_pol))/np.sum((matrizpopswap0.values))
        POPSWAPOUT= np.sum((matrizpopswap1.values*mx_pol_out))/np.sum((matrizpopswap1.values))
        POPININ   = np.sum((matrizpopswap1.values*mx_pol))/np.sum((matrizpopswap1.values))
        POPOUTOUT = np.sum((matrizpopswap0.values*mx_pol_out))/np.sum((matrizpopswap0.values))
        
        html_str = pd.DataFrame(np.array([BAD_IN,BAD_OUT,SWAPIN,SWAPOUT,ININ,OUTOUT])*100,index=['IN','OUT','SWAP_IN','SWAP_OUT','IN_IN','OUT_OUT'],columns=['%']).to_html()+\
        pd.DataFrame(np.array([POP_IN,POP_OUT,POPSWAPIN,POPSWAPOUT,POPININ,POPOUTOUT])*100,index=['POP. IN','POP. OUT','POP. SWAPIN','POP. SWAPOUT','POP. ININ','POP. OUTOUT'],columns=['%']).to_html()+\
        pd.DataFrame(mx_pol,columns=matrizbadfinal.columns,index=matrizbadfinal.index).style.background_gradient(cmap=my_cmap).render()
        #
        display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    else:
        html_str = pd.DataFrame(np.array([BAD_IN,BAD_OUT])*100,index=['IN','OUT'],columns=['%']).to_html()+\
        pd.DataFrame(np.array([POP_IN,POP_OUT])*100,index=['POP. IN','POP. OUT'],columns=['%']).to_html()+\
        pd.DataFrame(mx_pol,columns=matrizbadfinal.columns,index=matrizbadfinal.index).style.background_gradient(cmap=my_cmap).render()
        #
        display_html(html_str.replace('table','table style="display:inline"'),raw=True)
        
def show_pol_2(grid,matrizbadfinal,matrizbadpop,matrizpop,matrizbadswap0,matrizpopswap0,matrizbadswap1,matrizpopswap1,var_swap,my_cmap):
    #Calculando BadRate no ponto, Swapin, Swapout, etc...
    mx_pol = np.zeros((matrizbadfinal.shape))
    mx_pol_out = np.zeros((matrizbadfinal.shape))

    for i in range(matrizbadfinal.shape[0]):
        for j in range(matrizbadfinal.shape[1]):
            mx_pol[i,j] = grid[i+1,j+1].value
            mx_pol_out[i,j] = not grid[i+1,j+1].value

    #
    BAD_IN = np.sum((matrizbadpop.values*mx_pol))/np.sum((matrizpop.values*mx_pol))
    BAD_OUT = np.sum((matrizbadpop.values*mx_pol_out))/np.sum((matrizpop.values*mx_pol_out))
    
    POP_IN = np.sum((matrizpop.values*mx_pol))/np.sum(matrizpop.values)
    POP_OUT = np.sum((matrizpop.values*mx_pol_out))/np.sum(matrizpop.values)
    #calculando swapin
    if var_swap != None:
        SWAPIN = np.sum((matrizbadswap0.values*mx_pol))/np.sum((matrizpopswap0.values*mx_pol))
        SWAPOUT = np.sum((matrizbadswap1.values*mx_pol_out))/np.sum((matrizpopswap1.values*mx_pol_out))
        ININ = np.sum((matrizbadswap1.values*mx_pol))/np.sum((matrizpopswap1.values*mx_pol))
        OUTOUT = np.sum((matrizbadswap0.values*mx_pol_out))/np.sum((matrizpopswap0.values*mx_pol_out))
    
        POPSWAPIN = np.sum((matrizpopswap0.values*mx_pol))/np.sum((matrizpopswap0.values))
        POPSWAPOUT= np.sum((matrizpopswap1.values*mx_pol_out))/np.sum((matrizpopswap1.values))
        POPININ   = np.sum((matrizpopswap1.values*mx_pol))/np.sum((matrizpopswap1.values))
        POPOUTOUT = np.sum((matrizpopswap0.values*mx_pol_out))/np.sum((matrizpopswap0.values))
        
        html_str = pd.DataFrame(np.array([BAD_IN,BAD_OUT,SWAPIN,SWAPOUT,ININ,OUTOUT])*100,index=['IN','OUT','SWAP_IN','SWAP_OUT','IN_IN','OUT_OUT'],columns=['%']).to_html()+\
        pd.DataFrame(np.array([POP_IN,POP_OUT,POPSWAPIN,POPSWAPOUT,POPININ,POPOUTOUT])*100,index=['POP. IN','POP. OUT','POP. SWAPIN','POP. SWAPOUT','POP. ININ','POP. OUTOUT'],columns=['%']).to_html()+\
        pd.DataFrame(mx_pol,columns=matrizbadfinal.columns,index=matrizbadfinal.index).style.background_gradient(cmap=my_cmap).render()
        #
        display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    else:
        html_str = pd.DataFrame(np.array([BAD_IN,BAD_OUT])*100,index=['IN','OUT'],columns=['%']).to_html()+\
        pd.DataFrame(np.array([POP_IN,POP_OUT])*100,index=['POP. IN','POP. OUT'],columns=['%']).to_html()+\
        pd.DataFrame(mx_pol,columns=matrizbadfinal.columns,index=matrizbadfinal.index).style.background_gradient(cmap=my_cmap).render()
        #
        display_html(html_str.replace('table','table style="display:inline"'),raw=True)	
