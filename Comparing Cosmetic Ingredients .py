#!/usr/bin/env python
# coding: utf-8

# ## 1. Cosmetics, chemicals... it's complicated
# <p>Whenever I want to try a new cosmetic item, it's so difficult to choose. It's actually more than difficult. It's sometimes scary because new items that I've never tried end up giving me skin trouble. We know the information we need is on the back of each product, but it's really hard to interpret those ingredient lists unless you're a chemist. You may be able to relate to this situation.</p>
# <p><img src="https://assets.datacamp.com/production/project_695/img/image_1.png" style="width:600px;height:400px;"></p>
# <p>So instead of buying and hoping for the best, why don't we use data science to help us predict which products may be good fits for us? In this notebook, we are going to create a content-based recommendation system where the 'content' will be the chemical components of cosmetics. Specifically, we will process ingredient lists for 1472 cosmetics on Sephora via <a href="https://en.wikipedia.org/wiki/Word_embedding">word embedding</a>, then visualize ingredient similarity using a machine learning method called t-SNE and an interactive visualization library called Bokeh. Let's inspect our data first.</p>

# In[123]:


# Import libraries
import pandas as pd;
import numpy as np;
from sklearn.manifold import TSNE;


# Load the data
df = pd.read_csv("datasets/cosmetics.csv");

# Check the first five rows 
display(df.sample(5));

# Inspect the types of products
df.Label.value_counts();


# ## 2. Focus on one product category and one skin type
# <p>There are six categories of product in our data (<strong><em>moisturizers, cleansers, face masks, eye creams</em></strong>, and <strong><em>sun protection</em></strong>) and there are five different skin types (<strong><em>combination, dry, normal, oily</em></strong> and <strong><em>sensitive</em></strong>). Because individuals have different product needs as well as different skin types, let's set up our workflow so its outputs (a t-SNE model and a visualization of that model) can be customized. For the example in this notebook, let's focus in on moisturizers for those with dry skin by filtering the data accordingly.</p>

# In[125]:


# Filter for moisturizers
moisturizers = df[df['Label'] == 'Moisturizer']

# Filter for dry skin as well
moisturizers_dry = moisturizers[moisturizers['Dry'] == 1]

# Reset index
moisturizers_dry = moisturizers_dry.reset_index(drop = True)


# ## 3. Tokenizing the ingredients
# <p>To get to our end goal of comparing ingredients in each product, we first need to do some preprocessing tasks and bookkeeping of the actual words in each product's ingredients list. The first step will be tokenizing the list of ingredients in <code>Ingredients</code> column. After splitting them into tokens, we'll make a binary bag of words. Then we will create a dictionary with the tokens, <code>ingredient_idx</code>, which will have the following format:</p>
# <p>{ <strong><em>"ingredient"</em></strong>: index value, … }</p>

# In[127]:


# Initialize dictionary, list, and initial index
ingredient_idx = {}
corpus = []
idx = 0 

# For loop for tokenization 
for i in range(len(moisturizers_dry)):    
    ingredients = moisturizers_dry['Ingredients'][i]
    ingredients_lower = ingredients.lower()
    tokens = ingredients_lower.split(', ')
    corpus.append(tokens)
    for ingredient in tokens:
        if ingredient not in ingredient_idx:
            ingredient_idx[ingredient] = idx
            idx += 1
            
# Check the result 
print("The index for decyl oleate is", ingredient_idx['decyl oleate'])


# ## 4. Initializing a document-term matrix (DTM)
# <p>The next step is making a document-term matrix (DTM). Here each cosmetic product will correspond to a document, and each chemical composition will correspond to a term. This means we can think of the matrix as a <em>“cosmetic-ingredient”</em> matrix. The size of the matrix should be as the picture shown below.
# <img src="https://assets.datacamp.com/production/project_695/img/image_2.PNG" style="width:600px;height:250px;">
# To create this matrix, we'll first make an empty matrix filled with zeros. The length of the matrix is the total number of cosmetic products in the data. The width of the matrix is the total number of ingredients. After initializing this empty matrix, we'll fill it in the following tasks. </p>

# In[129]:


# Get the number of items and tokens 
M = len(moisturizers_dry)     
N = len(ingredient_idx)

# Initialize a matrix of zeros
A = np.zeros((M, N))
A.shape


# ## 5. Creating a counter function
# <p>Before we can fill the matrix, let's create a function to count the tokens (i.e., an ingredients list) for each row. Our end goal is to fill the matrix with 1 or 0: if an ingredient is in a cosmetic, the value is 1. If not, it remains 0. The name of this function, <code>oh_encoder</code>, will become clear next.</p>

# In[131]:


# Define the oh_encoder function
def oh_encoder(tokens):
    x = np.zeros(N)    
    for ingredient in tokens:
        # Get the index for each ingredient
        idx = ingredient_idx[ingredient]
        # Put 1 at the corresponding indices
        x[idx] = 1
    return x


# ## 6. The Cosmetic-Ingredient matrix!
# <p>Now we'll apply the <code>oh_encoder()</code> functon to the tokens in <code>corpus</code> and set the values at each row of this matrix. So the result will tell us what ingredients each item is composed of. For example, if a cosmetic item contains <em>water, niacin, decyl aleate</em> and <em>sh-polypeptide-1</em>, the outcome of this item will be as follows. 
# <img src="https://assets.datacamp.com/production/project_695/img/image_3.PNG" style="width:800px;height:400px;">
# This is what we called one-hot encoding. By encoding each ingredient in the items, the <em>Cosmetic-Ingredient</em> matrix will be filled with binary values. </p>

# In[133]:


# Make a document-term matrix
i = 0
for tokens in corpus:
    A[i, :] = oh_encoder(tokens)
    i += 1


# ## 7. Dimension reduction with t-SNE
# <p>The dimensions of the existing matrix is (190, 2233), which means there are 2233 features in our data. For visualization, we should downsize this into two dimensions. We'll use t-SNE for reducing the dimension of the data here.</p>
# <p><strong><a href="https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding">T-distributed Stochastic Neighbor Embedding (t-SNE)</a></strong> is a nonlinear dimensionality reduction technique that is well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions. Specifically, this technique can reduce the dimension of data while keeping the similarities between the instances. This enables us to make a plot on the coordinate plane, which can be said as vectorizing. All of these cosmetic items in our data will be vectorized into two-dimensional coordinates, and the distances between the points will indicate the similarities between the items. </p>

# In[135]:


# Dimension reduction with t-SNE
model = TSNE(n_components = 2, learning_rate = 200, random_state = 42)
tsne_features = model.fit_transform(A)

# Make X, Y columns 
moisturizers_dry['X'] = tsne_features[:, 0]
moisturizers_dry['Y'] = tsne_features[:, 1]


# ## 8. Let's map the items with Bokeh
# <p>We are now ready to start creating our plot. With the t-SNE values, we can plot all our items on the coordinate plane. And the coolest part here is that it will also show us the name, the brand, the price and the rank of each item. Let's make a scatter plot using Bokeh and add a hover tool to show that information. Note that we won't display the plot yet as we will make some more additions to it.</p>

# In[137]:


from bokeh.io import show, output_notebook, push_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
output_notebook()

# Make a source and a scatter plot
source = ColumnDataSource(moisturizers_dry)
plot = figure(x_axis_label = 'T-SNE 1',
              y_axis_label = 'T-SNE 2', 
              width = 500, height = 400)
plot.circle(x = 'X', 
            y = 'Y', 
            source = source, 
            size = 10, color = '#FF7373', alpha = .8)


# ## 9. Adding a hover tool
# <p>Why don't we add a hover tool? Adding a hover tool allows us to check the information of each item whenever the cursor is directly over a glyph. We'll add tooltips with each product's name, brand, price, and rank (i.e., rating).</p>

# In[139]:


# Create a HoverTool object
hover = HoverTool(tooltips = [('Item', '@Name'),
                              ('Brand', '@Brand'),
                              ('Price', '$@Price'),
                              ('Rank', '@Rank')])
plot.add_tools(hover)


# ## 10. Mapping the cosmetic items
# <p>Finally, it's show time! Let's see how the map we've made looks like. Each point on the plot corresponds to the cosmetic items. Then what do the axes mean here? The axes of a t-SNE plot aren't easily interpretable in terms of the original data. Like mentioned above, t-SNE is a visualizing technique to plot high-dimensional data in a low-dimensional space. Therefore, it's not desirable to interpret a t-SNE plot quantitatively.</p>
# <p>Instead, what we can get from this map is the distance between the points (which items are close and which are far apart). The closer the distance between the two items is, the more similar the composition they have. Therefore this enables us to compare the items without having any chemistry background.</p>

# In[141]:


# Plot the map
show(plot)


# ## 11. Comparing two products
# <p>Since there are so many cosmetics and so many ingredients, the plot doesn't have many super obvious patterns that simpler t-SNE plots can have (<a href="https://campus.datacamp.com/courses/unsupervised-learning-in-python/visualization-with-hierarchical-clustering-and-t-sne?ex=10">example</a>). Our plot requires some digging to find insights, but that's okay!</p>
# <p>Say we enjoyed a specific product, there's an increased chance we'd enjoy another product that is similar in chemical composition.  Say we enjoyed AmorePacific's <a href="https://www.sephora.com/product/color-control-cushion-compact-broad-spectrum-spf-50-P378121">Color Control Cushion Compact Broad Spectrum SPF 50+</a>. We could find this product on the plot and see if a similar product(s) exist. And it turns out it does! If we look at the points furthest left on the plot, we see  LANEIGE's <a href="https://www.sephora.com/product/bb-cushion-hydra-radiance-P420676">BB Cushion Hydra Radiance SPF 50</a> essentially overlaps with the AmorePacific product. By looking at the ingredients, we can visually confirm the compositions of the products are similar (<em>though it is difficult to do, which is why we did this analysis in the first place!</em>), plus LANEIGE's version is $22 cheaper and actually has higher ratings.</p>
# <p>It's not perfect, but it's useful. In real life, we can actually use our little ingredient-based recommendation engine help us make educated cosmetic purchase choices.</p>

# In[143]:


# Print the ingredients of two similar cosmetics
cosmetic_1 = moisturizers_dry[moisturizers_dry['Name'] == "Color Control Cushion Compact Broad Spectrum SPF 50+"]
cosmetic_2 = moisturizers_dry[moisturizers_dry['Name'] == "BB Cushion Hydra Radiance SPF 50"]

# Display each item's data and ingredients
display(cosmetic_1)
print(cosmetic_1.Ingredients.values)
display(cosmetic_2)
print(cosmetic_2.Ingredients.values)

