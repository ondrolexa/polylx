# -*- coding: utf-8 -*-
"""
Generate simple pdf reports.
Need rst2pdf tool (https://code.google.com/p/rst2pdf) to be installed.

Created on Wed Feb  5 21:42:54 2014

@author: Ondrej Lexa

Example:
  from polylx import *
  from polylx.reports import Report

  g = Grains.from_shp()

  fig, ax = plt.subplots()
  x = np.linspace(-8,8,200)
  ax.plot(x,np.sin(x))

  r = Report('Test report')
  r.add_chapter('Things will start here')
  r.savefig(fig, width='75%')
  r.table([[1,2,120],[2,6,213],[3,4,118]],
          title='Table example',
          header=['No','Val','Age'])
  r.grainmap(g, width='75%')
  r.write_pdf()

"""
from __future__ import print_function

import tempfile
import subprocess


class Report(object):

    def __init__(self, title="Report"):
        self.rst = []
        self.images = []
        poc = len(title)
        self.rst.append(poc*'=')
        self.rst.append(title)
        self.rst.append(poc*'=')
        self.fin()

    def fin(self):
        self.rst.append('')

    def add_chapter(self, title):
        poc = len(title)
        self.rst.append(title)
        self.rst.append(poc*'=')
        self.fin()

    def add_section(self, title):
        poc = len(title)
        self.rst.append(title)
        self.rst.append(poc*'-')
        self.fin()

    def add_subsection(self, title):
        poc = len(title)
        self.rst.append(title)
        self.rst.append(poc*'~')
        self.fin()

    def transition(self):
        self.rst.append('---------')
        self.fin()

    def figure(self, filename, width=None, height=None):
        self.rst.append('.. figure:: {}'.format(filename))
        if width:
            self.rst.append('   :width: {}'.format(width))
        if height:
            self.rst.append('   :height: {}'.format(height))
        self.fin()

    def matplotlib_fig(self, fig, width=None, height=None, bbox_inches='tight', dpi=150):
        f = tempfile.NamedTemporaryFile(suffix='.png')
        fig.savefig(f, format='png', bbox_inches=bbox_inches, dpi=dpi)
        self.figure(f.name, width, height)
        self.images.append(f)

    def plot(self, g, legend=None, loc='auto', alpha=0.8, dpi=150, width=None, height=None):
        f = tempfile.NamedTemporaryFile(suffix='.png')
        g.savefig(f, legend, loc, alpha, dpi)
        self.figure(f.name, width, height)
        self.images.append(f)

    def pagebreak(self):
        self.rst.append('.. raw:: pdf')
        self.fin()
        self.rst.append('   PageBreak')
        self.fin()

    def table(self, rows, title='Table', header=None, format=None, stub_columns=None, widths=None):
        self.rst.append('.. csv-table:: {}'.format(title))
        if header:
            self.rst.append('   :header: {}'.format(','.join(header)))
        if widths:
            self.rst.append('   :widths: {}'.format(','.join([str(w) for w in widths])))
        if stub_columns:
            self.rst.append('   :stub-columns: {}'.format(stub_columns))
        self.fin()
        if format is None:
            format = len(rows[0])*['']
        row_template = '   ' + ','.join(['{' + f + '}' for f in format])
        for row in rows:
            self.rst.append(row_template.format(*row))
        self.fin()

    def dataframe(self, df, title='Table', header=None, format=None, stub_columns=None, widths=None):
        rows = [row.split(',') for row in df.to_csv().split()]
        if not header:
            header = [df.index.name] + list(df.columns)
        rows = [[df.ix[i].name] + list(df.ix[i]) for i in range(len(df))]
        self.table(rows, title, header, format, stub_columns, widths)

    def write_rst(self, file='report.rst'):
        with open(file, 'w') as rstfile:
            for ln in self.rst:
                print(ln, file=rstfile)

    def write_pdf(self, file='report.pdf'):
        p = subprocess.Popen(['rst2pdf', '-s', 'dejavu', '-o', file],
                             stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        p.communicate(input='\n'.join(self.rst).encode(encoding='UTF-8'))
