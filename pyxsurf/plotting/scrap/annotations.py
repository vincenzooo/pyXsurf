import matplotlib.pyplot as plt
import textwrap as tw

plt.style.use('seaborn-notebook')
plt.rcParams['font.family'] = 'Ubuntu'

plt.figure(figsize=(8, 6), dpi=400)
plt.bar([1,2,3,4], [125,100,90,110], label="Product A", width=0.5,
            align='center')

# Text places anywhere within the Axis
plt.text(0.6, 130, 'Q1 accelerated with inventory refill',
         horizontalalignment='left', backgroundcolor='palegreen')

# The first line is escaped so that textwrap.dedent works correctly
comment1_txt = '''\
    Marketing campaign started in Q3 shows some impact in Q4. Further
    positive impact is expected in later quarters.
    '''
# We remove the indents and strip new lines from the text
# fill() creates the text with the specified width of 40 chars
annot_txt = tw.fill(tw.dedent(comment1_txt.rstrip()), width=40)

# Annotate using an altered arrowstyle for the head_width, the rest
# of the arguments are standard
plt.annotate(annot_txt, xy=(4,80), xytext=(1.50,105),
             arrowprops=dict(arrowstyle='-|>, head_width=0.5',
                             linewidth=2, facecolor='black'),
             bbox=dict(boxstyle="round", color='yellow', ec="0.5",
                       alpha=1), fontstyle='italic')

comment2_txt = '''\
    Notes: Sales for Product A have been flat through the year. We
    expect improvement after the new release in Q2.
    '''
fig_txt = tw.fill(tw.dedent(comment2_txt.rstrip() ), width=80)

# The YAxis value is -0.07 to push the text down slightly
plt.figtext(0.5, -0.07, fig_txt, horizontalalignment='center',
            fontsize=12, multialignment='left',
            bbox=dict(boxstyle="round", facecolor='#D8D8D8',
                      ec="0.5", pad=0.5, alpha=1), fontweight='bold')

# Standard description of the plot
plt.xticks([1,2,3,4],['Q1','Q2','Q3','Q4'])
plt.xlabel('Time')
plt.ylabel('Sales')
plt.title('Total sales by quarter', color='blue', fontstyle='italic')
plt.legend(loc='best')

plt.savefig('matplotlib-text-handling-example.svg', bbox_inches='tight')