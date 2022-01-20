<TeXmacs|2.1>

<style|generic>

<\body>
  <\big-table>
    <tabular|<tformat|<table|<row|<cell|attrs>|<cell|short name>|<cell|short
    desc.>|<cell|method>|<cell|backend>>|<row|<cell|<samp|box>>|<cell|>|<cell|with
    box bounds only>|<cell|<samp|BB>>|<cell|>>|<row|<cell|<samp|ball>>|<samp|TRS>|<cell|with
    ball bounds only (see TRS)>|<cell|<samp|BB>,
    <samp|SDR*>>|<cell|>>|<row|<cell|<samp|><samp|LC>>|<cell|<samp|LCQP>>|<cell|with
    linear constraints>|<cell|>|<cell|>>|<row|<cell|<samp|Q1>>|<cell|<samp|gTRS>
    >|<cell|with 1 nonconvex quadratic constraint>|<cell|<samp|SDR*>>|<cell|>>|<row|<cell|<samp|Q2>>|<cell|<samp|CDT>>|<cell|with
    2 quadratic constraint>|<cell|<samp|>>|<cell|>>|<row|<cell|<samp|Q2>>|<cell|<samp|gCDT>>|<cell|with
    2 nonconvex quadratic constraint>|<cell|>|<cell|>>|<row|<cell|<samp|QCQP-n<math|1>>>|<cell|>|<cell|with
    1 nonconvex eigenvalue>|<cell|>|<cell|>>|<row|<cell|<samp|QCQP-n<math|2>>>|<cell|>|<cell|with
    2 nonconvex eigenvalues>|<cell|>|<cell|>>|<row|<cell|<samp|><text-dots>>|<cell|>|<cell|>|<cell|>|<cell|>>>>>
  <|big-table>
    A classification of QCQP\ 
  </big-table>

  <\big-table|<tabular|<tformat|<table|<row|<cell|attr>|<cell|example>|<cell|short
  desc.>>|<row|<cell|<samp|box>>|<cell|<math|x\<in\><around*|[|l,u|]>>>|<cell|with
  box bounds>>|<row|<cell|<samp|ball>>|<cell|<math|x\<in\><around*|{|x:\<\|\|\>x\<\|\|\>\<leqslant\>\<delta\>|}>>>|<cell|with
  <math|\<cal-L\><rsub|2>> ball bounds>>|<row|<cell|<samp|n<math|>><math|1>,<samp|n><math|2>,<samp|n><math|r>>|<cell|>|<cell|with
  <math|1,2,r> nonconvex eigenvalue(s)>>|<row|<cell|<samp|Q<math|>><math|1>,Q<math|2>,<samp|Q><math|r>>|<cell|>|<cell|with
  <math|1,2,r> quadratic constraint(s)>>|<row|<cell|<samp|LC>>|<cell|<math|A
  x\<leqslant\>b>>|<cell|with linear constraints>>|<row|<cell|>|<cell|>|<cell|>>>>>>
    Different attributes of QCQP
  </big-table>
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1|../../../../../.TeXmacs/texts/scratch/no_name_4.tm>>
    <associate|auto-2|<tuple|2|?|../../../../../.TeXmacs/texts/scratch/no_name_4.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|table>
      <tuple|normal|<\surround|<hidden-binding|<tuple>|1>|>
        Classification of QCQP\ 
      </surround>|<pageref|auto-1>>
    </associate>
  </collection>
</auxiliary>