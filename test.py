
# coding: utf-8

# In[2]:


import numpy as np
import pydensecrf.densecrf as dcrf


# In[3]:


dcrf


# In[5]:


d = dcrf.DenseCRF2D(640, 480, 5)  # width, height, nlabels
d

