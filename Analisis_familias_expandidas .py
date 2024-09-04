#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importando paquetes
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# ### Extraccion de familias expandidas e contraidas

# In[7]:


# n de familias expandidas
lambdas=pd.read_csv("C://Users/wen/Downloads/results_lambda/Base_change.tab",sep ='\t')
#familias expandidas: A.tequilana
tequilana_expanded=lambdas[(lambdas['A_tequilana<2>']>=1)]
# familias expandidas:A.thaliana
thaliana_expanded=lambdas[(lambdas['A_thaliana<3>']>=1)]
#familias expandidas:Sugarcane
Sugarcane_expanded=lambdas[(lambdas['Sugarcane<1>']>=1)]


# In[11]:


# n de familias contraidas
lambdas=pd.read_csv("C://Users/wen/Downloads/results_lambda/Base_change.tab",sep ='\t')
#familias contraidas: A.tequilana
tequilana_contracted=lambdas[(lambdas['A_tequilana<2>']<=1)]
# familias contraidas:A.thaliana
thaliana_contracted=lambdas[(lambdas['A_thaliana<3>']<=1)]
#familias contraidas:Sugarcane
Sugarcane_contracted=lambdas[(lambdas['Sugarcane<1>']<=1)]


# ## Numero de familias expandidas (p <0.05)

# In[25]:


#familias significativas
lambdas_pvalue=pd.read_csv('C:/Users/wen/Downloads/results_lambda/Base_family_results.txt',sep='\t')#familias con pvalue <0.05:
lambdas_pvalue_005=lambdas_pvalue[(lambdas_pvalue.pvalue <=0.05) & (lambdas_pvalue ['Significant at 0.05']=="y")]


# In[31]:


#filtrar:familias expandidas A.tequilana <0.05
tequilana_expanded_pvalue005=tequilana_expanded[tequilana_expanded["FamilyID"].isin(list(lambdas_pvalue_005["#FamilyID"]))]
#filtrar:familias expandidas A.thaliana <0.05
thaliana_expanded_pvalue005=thaliana_expanded[thaliana_expanded["FamilyID"].isin(list(lambdas_pvalue_005["#FamilyID"]))]
##filtrar:familias expandidas Sugarcane0.05
Sugarcane_expanded_pvalue005=Sugarcane_expanded[Sugarcane_expanded["FamilyID"].isin(list(lambdas_pvalue_005["#FamilyID"]))]
#filtrar:familias contraidas A.tequilana <0.05
tequilana_contracted_pvalue005=tequilana_contracted[tequilana_contracted["FamilyID"].isin(list(lambdas_pvalue_005["#FamilyID"]))]
#filtrar:familias contraidas A.thaliana <0.05
thaliana_contracted_pvalue005=thaliana_contracted[thaliana_contracted["FamilyID"].isin(list(lambdas_pvalue_005["#FamilyID"]))]
#filtrar:familias contraidas Sugarcane <0.05
Sugarcane_contracted_pvalue005=Sugarcane_contracted[Sugarcane_contracted["FamilyID"].isin(list(lambdas_pvalue_005["#FamilyID"]))]


# In[48]:


#criando un diccionario expanded:
dic_lambdas_exp_tequilana={'lambda':['lambda_1',],
                          'Family_Sig':[len(tequilana_expanded_pvalue005)]}

dic_lambdas_exp_thaliana={'lambda':['lambda_1',],
                          'Family_Sig':[len(thaliana_expanded_pvalue005)]}

dic_lambdas_exp_sugarcane={'lambda':['lambda_1',],
                          'Family_Sig':[len(Sugarcane_expanded_pvalue005)]}
#criando un diccionario contracted
dic_lambdas_conc_tequilana={'lambda':['lambda_1',],
                          'Family_Sig':[len(tequilana_contracted_pvalue005)]}

dic_lambdas_conc_thaliana={'lambda':['lambda_1',],
                          'Family_Sig':[len(thaliana_contracted_pvalue005)]}

dic_lambdas_conc_sugarcane={'lambda':['lambda_1',],
                          'Family_Sig':[len(Sugarcane_contracted_pvalue005)]}


#criando un dataframe expanded
df_lamba_exp_tequilana=pd.DataFrame(data=dic_lambdas_exp_tequilana)
df_lamba_exp_thaliana=pd.DataFrame(data=dic_lambdas_exp_thaliana)
df_lamba_exp_sugarcane=pd.DataFrame(data=dic_lambdas_exp_sugarcane)
# crianda dataframe contracted
df_lamba_conc_tequilana=pd.DataFrame(data=dic_lambdas_conc_tequilana)
df_lamba_conc_thaliana=pd.DataFrame(data=dic_lambdas_conc_thaliana)
df_lamba_conc_sugarcane=pd.DataFrame(data=dic_lambdas_conc_sugarcane)


# In[38]:


df_lamba_exp_tequilana


# In[41]:


#criando diccionarios expanded 
dic_lambdas_fam_exp={'species':['A_tequilana<2>','A_thaliana<3>','Sugarcane<1>'],
                     'Family_Exp_Sig':[len(tequilana_expanded_pvalue005), 
                                       len (thaliana_expanded_pvalue005),
                                       len (Sugarcane_expanded_pvalue005)]}
df_lambdas_fam_exp=pd.DataFrame(data=dic_lambdas_fam_exp)


# In[42]:


df_lambdas_fam_exp


# In[44]:


# Grafico:
plt.figure(figsize = (10,7))
fam_expan_sig_grap = sns.barplot(data = df_lambdas_fam_exp, x = "species", y = "Family_Exp_Sig")
fam_expan_sig_grap.set_xticklabels(fam_expan_sig_grap.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
# Personalizcao:
fam_expan_sig_grap.set_xlabel("", fontsize = 50)
fam_expan_sig_grap.set_ylabel("# extended families", fontsize = 20)
plt.title('Extended families - lambda; p-value < 0.05', fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

# Anotacao:
for p in fam_expan_sig_grap.patches:
    fam_expan_sig_grap.annotate("{:,.0f}".format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    


# In[46]:


#criando diccionarios expanded 
dic_lambdas_fam_conc={'species':['A_tequilana<2>','A_thaliana<3>','Sugarcane<1>'],
                     'Family_Conc_Sig':[len(tequilana_contracted_pvalue005), 
                                       len (thaliana_contracted_pvalue005),
                                       len (Sugarcane_contracted_pvalue005)]}
df_lambdas_fam_conc=pd.DataFrame(data=dic_lambdas_fam_conc)


# In[47]:


#Grafico:
plt.figure(figsize = (10,7))
fam_expan_sig_grap = sns.barplot(data = df_lambdas_fam_conc, x = "species", y = "Family_Conc_Sig")
fam_expan_sig_grap.set_xticklabels(fam_expan_sig_grap.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
# Personalizcao:
fam_expan_sig_grap.set_xlabel("", fontsize = 50)
fam_expan_sig_grap.set_ylabel("# extended families", fontsize = 20)
plt.title('Contracted families - lambda; p-value < 0.05', fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

# Anotacao:
for p in fam_expan_sig_grap.patches:
   fam_expan_sig_grap.annotate("{:,.0f}".format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
   


# ## UpSet Familias compartidas y exclusivas

# In[54]:


pip install UpSetPlot


# In[50]:


import numpy as np
# lambdas
# Selecioanr colunas:
lambdas_ft = lambdas[['FamilyID', 'A_tequilana<2>','A_thaliana<3>','Sugarcane<1>']]
# Altearando nome das linhas:
lambdas_ft.index = lambdas_ft['FamilyID']
# Deletar coluna:
lambdas_ft.drop('FamilyID', inplace=True, axis=1)
# Copiando dataframe:
lambdas_ft_exp = lambdas_ft.copy()
lambdas_ft_cont = lambdas_ft.copy()
# Substituindo todos os valores de um dataframe:
lambdas_ft_exp = lambdas_ft_exp.applymap(lambda x: 1 if x >= 1 else 0) # else x
lambdas_ft_cont = lambdas_ft_cont.applymap(lambda x: 1 if x <= -1 else 0) # else x
# Idex para coluna:
lambdas_ft_exp['#FamilyID'] = lambdas_ft_exp.index
lambdas_ft_cont['#FamilyID'] = lambdas_ft_cont.index
# Merge:
lambdas_ft_exp_pvalue = pd.merge(lambdas_ft_exp, lambdas_pvalue_005, on = "#FamilyID", how = 'inner') # how = 'left'
lambdas_ft_cont_pvalue  = pd.merge(lambdas_ft_exp, lambdas_pvalue_005, on = "#FamilyID", how = 'inner') # how = 'left'
# print("Familias expand: ", len(lambdas_ft_exp))
# print("Familias expand com p-value < 0.05: ", len(lambdas_ft_exp_pvalue))
# print("Familias contracted: ", len(lambdas_ft_cont))
# print("Familias contracted com p-value < 0.05: ", len(lambdas_ft_cont_pvalue))
# gamma_3_change_ft_cont_pvalue.head(2)
# Filtrado:
A_tequilana_df = lambdas_ft_exp_pvalue[(lambdas_ft_exp_pvalue['A_tequilana<2>'] == 1)]
A_thaliana_df = lambdas_ft_exp_pvalue[(lambdas_ft_exp_pvalue['A_thaliana<3>'] == 1)]
Sugarcane_df = lambdas_ft_exp_pvalue[(lambdas_ft_exp_pvalue['Sugarcane<1>'] == 1)]


# Verificar:
# gamma_3_fam_exp_erinaceum_005


# In[55]:



# Obetendo lista das familias:
set_exp_tequilana_1= set(tequilana_expanded_pvalue005['FamilyID'])
set_exp_thaliana_2 = set(thaliana_expanded_pvalue005['FamilyID'])
set_exp_sugarcane_3 = set(Sugarcane_expanded_pvalue005['FamilyID'])

# Grafico:
set_names = ['A_tequilana', 'A_thaliana','sugarcane']
all_elems = set_exp_tequilana_1.union(set_exp_thaliana_2).union(set_exp_sugarcane_3)
df = pd.DataFrame([[e in set_exp_tequilana_1, e in set_exp_thaliana_2, e in set_exp_sugarcane_3] for e in all_elems], columns = set_names)
df_up = df.groupby(set_names).size()

# Figura:
from upsetplot import plot
fig = plt.figure(figsize=(16, 8))
plot(df_up, orientation = 'horizontal', fig = fig, element_size = None, show_counts = True)
plt.show()

