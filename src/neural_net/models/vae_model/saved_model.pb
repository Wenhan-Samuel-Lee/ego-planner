��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*1.14.02v1.14.0-rc1-22-gaf24dc91b58��
�
conv2d_16/kernelVarHandleOp*
dtype0*
_output_shapes
: *!
shared_nameconv2d_16/kernel*
shape: 
�
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*
dtype0*#
_class
loc:@conv2d_16/kernel*&
_output_shapes
: 
t
conv2d_16/biasVarHandleOp*
shared_nameconv2d_16/bias*
_output_shapes
: *
shape: *
dtype0
�
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*!
_class
loc:@conv2d_16/bias*
_output_shapes
: *
dtype0
�
conv2d_17/kernelVarHandleOp*!
shared_nameconv2d_17/kernel*
dtype0*
_output_shapes
: *
shape:  
�
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*#
_class
loc:@conv2d_17/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_17/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_17/bias*
shape: 
�
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*!
_class
loc:@conv2d_17/bias*
_output_shapes
: *
dtype0
�
conv2d_18/kernelVarHandleOp*
_output_shapes
: *
shape: @*!
shared_nameconv2d_18/kernel*
dtype0
�
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
: @*
dtype0*#
_class
loc:@conv2d_18/kernel
t
conv2d_18/biasVarHandleOp*
shape:@*
shared_nameconv2d_18/bias*
dtype0*
_output_shapes
: 
�
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*!
_class
loc:@conv2d_18/bias*
dtype0*
_output_shapes
:@
�
conv2d_19/kernelVarHandleOp*
_output_shapes
: *
shape:@@*!
shared_nameconv2d_19/kernel*
dtype0
�
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*&
_output_shapes
:@@*
dtype0*#
_class
loc:@conv2d_19/kernel
t
conv2d_19/biasVarHandleOp*
dtype0*
shared_nameconv2d_19/bias*
shape:@*
_output_shapes
: 
�
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
dtype0*
_output_shapes
:@*!
_class
loc:@conv2d_19/bias
|
z_mean_4/kernelVarHandleOp*
_output_shapes
: * 
shared_namez_mean_4/kernel*
dtype0*
shape:
��
�
#z_mean_4/kernel/Read/ReadVariableOpReadVariableOpz_mean_4/kernel* 
_output_shapes
:
��*"
_class
loc:@z_mean_4/kernel*
dtype0
s
z_mean_4/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_namez_mean_4/bias
�
!z_mean_4/bias/Read/ReadVariableOpReadVariableOpz_mean_4/bias*
_output_shapes	
:�*
dtype0* 
_class
loc:@z_mean_4/bias
�
z_log_var_4/kernelVarHandleOp*#
shared_namez_log_var_4/kernel*
dtype0*
_output_shapes
: *
shape:
��
�
&z_log_var_4/kernel/Read/ReadVariableOpReadVariableOpz_log_var_4/kernel* 
_output_shapes
:
��*
dtype0*%
_class
loc:@z_log_var_4/kernel
y
z_log_var_4/biasVarHandleOp*
dtype0*!
shared_namez_log_var_4/bias*
shape:�*
_output_shapes
: 
�
$z_log_var_4/bias/Read/ReadVariableOpReadVariableOpz_log_var_4/bias*#
_class
loc:@z_log_var_4/bias*
dtype0*
_output_shapes	
:�
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
shape:
��*
dtype0*
shared_namedense_4/kernel
�
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
��*
dtype0*!
_class
loc:@dense_4/kernel
q
dense_4/biasVarHandleOp*
shared_namedense_4/bias*
dtype0*
shape:�*
_output_shapes
: 
�
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:�*
_class
loc:@dense_4/bias*
dtype0
�
conv2d_transpose_16/kernelVarHandleOp*+
shared_nameconv2d_transpose_16/kernel*
shape:@@*
dtype0*
_output_shapes
: 
�
.conv2d_transpose_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_16/kernel*&
_output_shapes
:@@*-
_class#
!loc:@conv2d_transpose_16/kernel*
dtype0
�
conv2d_transpose_16/biasVarHandleOp*
dtype0*)
shared_nameconv2d_transpose_16/bias*
_output_shapes
: *
shape:@
�
,conv2d_transpose_16/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_16/bias*
dtype0*
_output_shapes
:@*+
_class!
loc:@conv2d_transpose_16/bias
�
conv2d_transpose_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_17/kernel
�
.conv2d_transpose_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_17/kernel*&
_output_shapes
: @*-
_class#
!loc:@conv2d_transpose_17/kernel*
dtype0
�
conv2d_transpose_17/biasVarHandleOp*
shape: *)
shared_nameconv2d_transpose_17/bias*
dtype0*
_output_shapes
: 
�
,conv2d_transpose_17/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_17/bias*
_output_shapes
: *
dtype0*+
_class!
loc:@conv2d_transpose_17/bias
�
conv2d_transpose_18/kernelVarHandleOp*
shape:  *
_output_shapes
: *
dtype0*+
shared_nameconv2d_transpose_18/kernel
�
.conv2d_transpose_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_18/kernel*
dtype0*&
_output_shapes
:  *-
_class#
!loc:@conv2d_transpose_18/kernel
�
conv2d_transpose_18/biasVarHandleOp*
shape: *
dtype0*
_output_shapes
: *)
shared_nameconv2d_transpose_18/bias
�
,conv2d_transpose_18/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_18/bias*
dtype0*
_output_shapes
: *+
_class!
loc:@conv2d_transpose_18/bias
�
conv2d_transpose_19/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape: *+
shared_nameconv2d_transpose_19/kernel
�
.conv2d_transpose_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_19/kernel*
dtype0*&
_output_shapes
: *-
_class#
!loc:@conv2d_transpose_19/kernel
�
conv2d_transpose_19/biasVarHandleOp*
shape:*)
shared_nameconv2d_transpose_19/bias*
_output_shapes
: *
dtype0
�
,conv2d_transpose_19/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_19/bias*
dtype0*+
_class!
loc:@conv2d_transpose_19/bias*
_output_shapes
:
j
Adam_4/iterVarHandleOp*
shared_nameAdam_4/iter*
dtype0	*
_output_shapes
: *
shape: 
�
Adam_4/iter/Read/ReadVariableOpReadVariableOpAdam_4/iter*
dtype0	*
_output_shapes
: *
_class
loc:@Adam_4/iter
n
Adam_4/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam_4/beta_1
�
!Adam_4/beta_1/Read/ReadVariableOpReadVariableOpAdam_4/beta_1* 
_class
loc:@Adam_4/beta_1*
dtype0*
_output_shapes
: 
n
Adam_4/beta_2VarHandleOp*
_output_shapes
: *
shape: *
shared_nameAdam_4/beta_2*
dtype0
�
!Adam_4/beta_2/Read/ReadVariableOpReadVariableOpAdam_4/beta_2* 
_class
loc:@Adam_4/beta_2*
_output_shapes
: *
dtype0
l
Adam_4/decayVarHandleOp*
_output_shapes
: *
shape: *
shared_nameAdam_4/decay*
dtype0
�
 Adam_4/decay/Read/ReadVariableOpReadVariableOpAdam_4/decay*
_class
loc:@Adam_4/decay*
dtype0*
_output_shapes
: 
|
Adam_4/learning_rateVarHandleOp*
dtype0*%
shared_nameAdam_4/learning_rate*
_output_shapes
: *
shape: 
�
(Adam_4/learning_rate/Read/ReadVariableOpReadVariableOpAdam_4/learning_rate*
dtype0*
_output_shapes
: *'
_class
loc:@Adam_4/learning_rate
�
Adam_4/conv2d_16/kernel/mVarHandleOp**
shared_nameAdam_4/conv2d_16/kernel/m*
_output_shapes
: *
dtype0*
shape: 
�
-Adam_4/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_16/kernel/m*
dtype0*,
_class"
 loc:@Adam_4/conv2d_16/kernel/m*&
_output_shapes
: 
�
Adam_4/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
shape: *
dtype0*(
shared_nameAdam_4/conv2d_16/bias/m
�
+Adam_4/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_16/bias/m*
_output_shapes
: **
_class 
loc:@Adam_4/conv2d_16/bias/m*
dtype0
�
Adam_4/conv2d_17/kernel/mVarHandleOp**
shared_nameAdam_4/conv2d_17/kernel/m*
shape:  *
_output_shapes
: *
dtype0
�
-Adam_4/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_17/kernel/m*&
_output_shapes
:  *
dtype0*,
_class"
 loc:@Adam_4/conv2d_17/kernel/m
�
Adam_4/conv2d_17/bias/mVarHandleOp*
dtype0*
shape: *(
shared_nameAdam_4/conv2d_17/bias/m*
_output_shapes
: 
�
+Adam_4/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_17/bias/m*
dtype0**
_class 
loc:@Adam_4/conv2d_17/bias/m*
_output_shapes
: 
�
Adam_4/conv2d_18/kernel/mVarHandleOp**
shared_nameAdam_4/conv2d_18/kernel/m*
_output_shapes
: *
dtype0*
shape: @
�
-Adam_4/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_18/kernel/m*,
_class"
 loc:@Adam_4/conv2d_18/kernel/m*
dtype0*&
_output_shapes
: @
�
Adam_4/conv2d_18/bias/mVarHandleOp*
dtype0*(
shared_nameAdam_4/conv2d_18/bias/m*
shape:@*
_output_shapes
: 
�
+Adam_4/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_18/bias/m**
_class 
loc:@Adam_4/conv2d_18/bias/m*
dtype0*
_output_shapes
:@
�
Adam_4/conv2d_19/kernel/mVarHandleOp*
shape:@@*
dtype0*
_output_shapes
: **
shared_nameAdam_4/conv2d_19/kernel/m
�
-Adam_4/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_19/kernel/m*&
_output_shapes
:@@*,
_class"
 loc:@Adam_4/conv2d_19/kernel/m*
dtype0
�
Adam_4/conv2d_19/bias/mVarHandleOp*(
shared_nameAdam_4/conv2d_19/bias/m*
_output_shapes
: *
dtype0*
shape:@
�
+Adam_4/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_19/bias/m*
dtype0*
_output_shapes
:@**
_class 
loc:@Adam_4/conv2d_19/bias/m
�
Adam_4/z_mean_4/kernel/mVarHandleOp*
shape:
��*
_output_shapes
: *
dtype0*)
shared_nameAdam_4/z_mean_4/kernel/m
�
,Adam_4/z_mean_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam_4/z_mean_4/kernel/m* 
_output_shapes
:
��*+
_class!
loc:@Adam_4/z_mean_4/kernel/m*
dtype0
�
Adam_4/z_mean_4/bias/mVarHandleOp*'
shared_nameAdam_4/z_mean_4/bias/m*
_output_shapes
: *
dtype0*
shape:�
�
*Adam_4/z_mean_4/bias/m/Read/ReadVariableOpReadVariableOpAdam_4/z_mean_4/bias/m*
dtype0*)
_class
loc:@Adam_4/z_mean_4/bias/m*
_output_shapes	
:�
�
Adam_4/z_log_var_4/kernel/mVarHandleOp*
_output_shapes
: *
shape:
��*,
shared_nameAdam_4/z_log_var_4/kernel/m*
dtype0
�
/Adam_4/z_log_var_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam_4/z_log_var_4/kernel/m* 
_output_shapes
:
��*.
_class$
" loc:@Adam_4/z_log_var_4/kernel/m*
dtype0
�
Adam_4/z_log_var_4/bias/mVarHandleOp*
_output_shapes
: *
shape:�*
dtype0**
shared_nameAdam_4/z_log_var_4/bias/m
�
-Adam_4/z_log_var_4/bias/m/Read/ReadVariableOpReadVariableOpAdam_4/z_log_var_4/bias/m*,
_class"
 loc:@Adam_4/z_log_var_4/bias/m*
_output_shapes	
:�*
dtype0
�
Adam_4/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_nameAdam_4/dense_4/kernel/m
�
+Adam_4/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam_4/dense_4/kernel/m* 
_output_shapes
:
��*
dtype0**
_class 
loc:@Adam_4/dense_4/kernel/m
�
Adam_4/dense_4/bias/mVarHandleOp*
dtype0*&
shared_nameAdam_4/dense_4/bias/m*
_output_shapes
: *
shape:�
�
)Adam_4/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam_4/dense_4/bias/m*
_output_shapes	
:�*
dtype0*(
_class
loc:@Adam_4/dense_4/bias/m
�
#Adam_4/conv2d_transpose_16/kernel/mVarHandleOp*
shape:@@*4
shared_name%#Adam_4/conv2d_transpose_16/kernel/m*
_output_shapes
: *
dtype0
�
7Adam_4/conv2d_transpose_16/kernel/m/Read/ReadVariableOpReadVariableOp#Adam_4/conv2d_transpose_16/kernel/m*&
_output_shapes
:@@*6
_class,
*(loc:@Adam_4/conv2d_transpose_16/kernel/m*
dtype0
�
!Adam_4/conv2d_transpose_16/bias/mVarHandleOp*
shape:@*
_output_shapes
: *
dtype0*2
shared_name#!Adam_4/conv2d_transpose_16/bias/m
�
5Adam_4/conv2d_transpose_16/bias/m/Read/ReadVariableOpReadVariableOp!Adam_4/conv2d_transpose_16/bias/m*
dtype0*
_output_shapes
:@*4
_class*
(&loc:@Adam_4/conv2d_transpose_16/bias/m
�
#Adam_4/conv2d_transpose_17/kernel/mVarHandleOp*
dtype0*4
shared_name%#Adam_4/conv2d_transpose_17/kernel/m*
_output_shapes
: *
shape: @
�
7Adam_4/conv2d_transpose_17/kernel/m/Read/ReadVariableOpReadVariableOp#Adam_4/conv2d_transpose_17/kernel/m*6
_class,
*(loc:@Adam_4/conv2d_transpose_17/kernel/m*&
_output_shapes
: @*
dtype0
�
!Adam_4/conv2d_transpose_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*2
shared_name#!Adam_4/conv2d_transpose_17/bias/m*
shape: 
�
5Adam_4/conv2d_transpose_17/bias/m/Read/ReadVariableOpReadVariableOp!Adam_4/conv2d_transpose_17/bias/m*
_output_shapes
: *
dtype0*4
_class*
(&loc:@Adam_4/conv2d_transpose_17/bias/m
�
#Adam_4/conv2d_transpose_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *4
shared_name%#Adam_4/conv2d_transpose_18/kernel/m
�
7Adam_4/conv2d_transpose_18/kernel/m/Read/ReadVariableOpReadVariableOp#Adam_4/conv2d_transpose_18/kernel/m*6
_class,
*(loc:@Adam_4/conv2d_transpose_18/kernel/m*
dtype0*&
_output_shapes
:  
�
!Adam_4/conv2d_transpose_18/bias/mVarHandleOp*2
shared_name#!Adam_4/conv2d_transpose_18/bias/m*
shape: *
dtype0*
_output_shapes
: 
�
5Adam_4/conv2d_transpose_18/bias/m/Read/ReadVariableOpReadVariableOp!Adam_4/conv2d_transpose_18/bias/m*
dtype0*4
_class*
(&loc:@Adam_4/conv2d_transpose_18/bias/m*
_output_shapes
: 
�
#Adam_4/conv2d_transpose_19/kernel/mVarHandleOp*4
shared_name%#Adam_4/conv2d_transpose_19/kernel/m*
shape: *
_output_shapes
: *
dtype0
�
7Adam_4/conv2d_transpose_19/kernel/m/Read/ReadVariableOpReadVariableOp#Adam_4/conv2d_transpose_19/kernel/m*&
_output_shapes
: *
dtype0*6
_class,
*(loc:@Adam_4/conv2d_transpose_19/kernel/m
�
!Adam_4/conv2d_transpose_19/bias/mVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!Adam_4/conv2d_transpose_19/bias/m*
shape:
�
5Adam_4/conv2d_transpose_19/bias/m/Read/ReadVariableOpReadVariableOp!Adam_4/conv2d_transpose_19/bias/m*4
_class*
(&loc:@Adam_4/conv2d_transpose_19/bias/m*
dtype0*
_output_shapes
:
�
Adam_4/conv2d_16/kernel/vVarHandleOp**
shared_nameAdam_4/conv2d_16/kernel/v*
shape: *
_output_shapes
: *
dtype0
�
-Adam_4/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_16/kernel/v*,
_class"
 loc:@Adam_4/conv2d_16/kernel/v*&
_output_shapes
: *
dtype0
�
Adam_4/conv2d_16/bias/vVarHandleOp*
shape: *
dtype0*
_output_shapes
: *(
shared_nameAdam_4/conv2d_16/bias/v
�
+Adam_4/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_16/bias/v*
dtype0*
_output_shapes
: **
_class 
loc:@Adam_4/conv2d_16/bias/v
�
Adam_4/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
shape:  **
shared_nameAdam_4/conv2d_17/kernel/v*
dtype0
�
-Adam_4/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_17/kernel/v*
dtype0*,
_class"
 loc:@Adam_4/conv2d_17/kernel/v*&
_output_shapes
:  
�
Adam_4/conv2d_17/bias/vVarHandleOp*
dtype0*
shape: *
_output_shapes
: *(
shared_nameAdam_4/conv2d_17/bias/v
�
+Adam_4/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_17/bias/v*
_output_shapes
: **
_class 
loc:@Adam_4/conv2d_17/bias/v*
dtype0
�
Adam_4/conv2d_18/kernel/vVarHandleOp*
dtype0*
shape: @**
shared_nameAdam_4/conv2d_18/kernel/v*
_output_shapes
: 
�
-Adam_4/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_18/kernel/v*,
_class"
 loc:@Adam_4/conv2d_18/kernel/v*&
_output_shapes
: @*
dtype0
�
Adam_4/conv2d_18/bias/vVarHandleOp*(
shared_nameAdam_4/conv2d_18/bias/v*
_output_shapes
: *
dtype0*
shape:@
�
+Adam_4/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_18/bias/v*
_output_shapes
:@**
_class 
loc:@Adam_4/conv2d_18/bias/v*
dtype0
�
Adam_4/conv2d_19/kernel/vVarHandleOp*
shape:@@*
dtype0**
shared_nameAdam_4/conv2d_19/kernel/v*
_output_shapes
: 
�
-Adam_4/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_19/kernel/v*,
_class"
 loc:@Adam_4/conv2d_19/kernel/v*
dtype0*&
_output_shapes
:@@
�
Adam_4/conv2d_19/bias/vVarHandleOp*
shape:@*
_output_shapes
: *(
shared_nameAdam_4/conv2d_19/bias/v*
dtype0
�
+Adam_4/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam_4/conv2d_19/bias/v*
dtype0**
_class 
loc:@Adam_4/conv2d_19/bias/v*
_output_shapes
:@
�
Adam_4/z_mean_4/kernel/vVarHandleOp*
_output_shapes
: *
shape:
��*)
shared_nameAdam_4/z_mean_4/kernel/v*
dtype0
�
,Adam_4/z_mean_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam_4/z_mean_4/kernel/v*
dtype0*+
_class!
loc:@Adam_4/z_mean_4/kernel/v* 
_output_shapes
:
��
�
Adam_4/z_mean_4/bias/vVarHandleOp*
shape:�*'
shared_nameAdam_4/z_mean_4/bias/v*
_output_shapes
: *
dtype0
�
*Adam_4/z_mean_4/bias/v/Read/ReadVariableOpReadVariableOpAdam_4/z_mean_4/bias/v*
dtype0*)
_class
loc:@Adam_4/z_mean_4/bias/v*
_output_shapes	
:�
�
Adam_4/z_log_var_4/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*,
shared_nameAdam_4/z_log_var_4/kernel/v
�
/Adam_4/z_log_var_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam_4/z_log_var_4/kernel/v*
dtype0*.
_class$
" loc:@Adam_4/z_log_var_4/kernel/v* 
_output_shapes
:
��
�
Adam_4/z_log_var_4/bias/vVarHandleOp**
shared_nameAdam_4/z_log_var_4/bias/v*
dtype0*
shape:�*
_output_shapes
: 
�
-Adam_4/z_log_var_4/bias/v/Read/ReadVariableOpReadVariableOpAdam_4/z_log_var_4/bias/v*,
_class"
 loc:@Adam_4/z_log_var_4/bias/v*
dtype0*
_output_shapes	
:�
�
Adam_4/dense_4/kernel/vVarHandleOp*
shape:
��*
dtype0*
_output_shapes
: *(
shared_nameAdam_4/dense_4/kernel/v
�
+Adam_4/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam_4/dense_4/kernel/v* 
_output_shapes
:
��**
_class 
loc:@Adam_4/dense_4/kernel/v*
dtype0
�
Adam_4/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*&
shared_nameAdam_4/dense_4/bias/v*
shape:�
�
)Adam_4/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam_4/dense_4/bias/v*
_output_shapes	
:�*(
_class
loc:@Adam_4/dense_4/bias/v*
dtype0
�
#Adam_4/conv2d_transpose_16/kernel/vVarHandleOp*
shape:@@*4
shared_name%#Adam_4/conv2d_transpose_16/kernel/v*
_output_shapes
: *
dtype0
�
7Adam_4/conv2d_transpose_16/kernel/v/Read/ReadVariableOpReadVariableOp#Adam_4/conv2d_transpose_16/kernel/v*6
_class,
*(loc:@Adam_4/conv2d_transpose_16/kernel/v*&
_output_shapes
:@@*
dtype0
�
!Adam_4/conv2d_transpose_16/bias/vVarHandleOp*
_output_shapes
: *2
shared_name#!Adam_4/conv2d_transpose_16/bias/v*
shape:@*
dtype0
�
5Adam_4/conv2d_transpose_16/bias/v/Read/ReadVariableOpReadVariableOp!Adam_4/conv2d_transpose_16/bias/v*
_output_shapes
:@*
dtype0*4
_class*
(&loc:@Adam_4/conv2d_transpose_16/bias/v
�
#Adam_4/conv2d_transpose_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*4
shared_name%#Adam_4/conv2d_transpose_17/kernel/v*
shape: @
�
7Adam_4/conv2d_transpose_17/kernel/v/Read/ReadVariableOpReadVariableOp#Adam_4/conv2d_transpose_17/kernel/v*6
_class,
*(loc:@Adam_4/conv2d_transpose_17/kernel/v*&
_output_shapes
: @*
dtype0
�
!Adam_4/conv2d_transpose_17/bias/vVarHandleOp*
shape: *
_output_shapes
: *
dtype0*2
shared_name#!Adam_4/conv2d_transpose_17/bias/v
�
5Adam_4/conv2d_transpose_17/bias/v/Read/ReadVariableOpReadVariableOp!Adam_4/conv2d_transpose_17/bias/v*
_output_shapes
: *4
_class*
(&loc:@Adam_4/conv2d_transpose_17/bias/v*
dtype0
�
#Adam_4/conv2d_transpose_18/kernel/vVarHandleOp*
_output_shapes
: *4
shared_name%#Adam_4/conv2d_transpose_18/kernel/v*
shape:  *
dtype0
�
7Adam_4/conv2d_transpose_18/kernel/v/Read/ReadVariableOpReadVariableOp#Adam_4/conv2d_transpose_18/kernel/v*&
_output_shapes
:  *
dtype0*6
_class,
*(loc:@Adam_4/conv2d_transpose_18/kernel/v
�
!Adam_4/conv2d_transpose_18/bias/vVarHandleOp*
shape: *
dtype0*
_output_shapes
: *2
shared_name#!Adam_4/conv2d_transpose_18/bias/v
�
5Adam_4/conv2d_transpose_18/bias/v/Read/ReadVariableOpReadVariableOp!Adam_4/conv2d_transpose_18/bias/v*4
_class*
(&loc:@Adam_4/conv2d_transpose_18/bias/v*
dtype0*
_output_shapes
: 
�
#Adam_4/conv2d_transpose_19/kernel/vVarHandleOp*
dtype0*4
shared_name%#Adam_4/conv2d_transpose_19/kernel/v*
_output_shapes
: *
shape: 
�
7Adam_4/conv2d_transpose_19/kernel/v/Read/ReadVariableOpReadVariableOp#Adam_4/conv2d_transpose_19/kernel/v*
dtype0*6
_class,
*(loc:@Adam_4/conv2d_transpose_19/kernel/v*&
_output_shapes
: 
�
!Adam_4/conv2d_transpose_19/bias/vVarHandleOp*
dtype0*
shape:*2
shared_name#!Adam_4/conv2d_transpose_19/bias/v*
_output_shapes
: 
�
5Adam_4/conv2d_transpose_19/bias/v/Read/ReadVariableOpReadVariableOp!Adam_4/conv2d_transpose_19/bias/v*4
_class*
(&loc:@Adam_4/conv2d_transpose_19/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
�L
ConstConst"/device:CPU:0*�K
value�KB�K B�K
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
	optimizer

signatures
 


kernel
bias


kernel
bias


kernel
bias


kernel
bias
 


kernel
bias


kernel
bias
 


kernel
bias
 


 kernel
!bias


"kernel
#bias


$kernel
%bias


&kernel
'bias
�
(iter

)beta_1

*beta_2
	+decay
,learning_ratem-m.m/m0m1m2m3m4m5m6m7m8m9m: m;!m<"m=#m>$m?%m@&mA'mBvCvDvEvFvGvHvIvJvKvLvMvNvOvP vQ!vR"vS#vT$vU%vV&vW'vX
 
\Z
VARIABLE_VALUEconv2d_16/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_16/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv2d_17/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_17/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEz_mean_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEz_mean_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEz_log_var_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEz_log_var_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEconv2d_transpose_16/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_16/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEconv2d_transpose_17/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_17/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEconv2d_transpose_18/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_18/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEconv2d_transpose_19/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEconv2d_transpose_19/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdam_4/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdam_4/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdam_4/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam_4/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEAdam_4/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam_4/conv2d_16/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam_4/conv2d_16/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam_4/conv2d_17/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam_4/conv2d_17/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam_4/conv2d_18/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam_4/conv2d_18/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam_4/conv2d_19/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam_4/conv2d_19/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam_4/z_mean_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam_4/z_mean_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam_4/z_log_var_4/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam_4/z_log_var_4/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam_4/dense_4/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam_4/dense_4/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam_4/conv2d_transpose_16/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam_4/conv2d_transpose_16/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam_4/conv2d_transpose_17/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam_4/conv2d_transpose_17/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam_4/conv2d_transpose_18/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam_4/conv2d_transpose_18/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam_4/conv2d_transpose_19/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam_4/conv2d_transpose_19/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam_4/conv2d_16/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam_4/conv2d_16/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam_4/conv2d_17/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam_4/conv2d_17/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam_4/conv2d_18/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam_4/conv2d_18/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam_4/conv2d_19/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam_4/conv2d_19/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam_4/z_mean_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam_4/z_mean_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam_4/z_log_var_4/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam_4/z_log_var_4/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam_4/dense_4/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam_4/dense_4/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam_4/conv2d_transpose_16/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam_4/conv2d_transpose_16/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam_4/conv2d_transpose_17/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam_4/conv2d_transpose_17/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam_4/conv2d_transpose_18/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam_4/conv2d_transpose_18/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam_4/conv2d_transpose_19/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam_4/conv2d_transpose_19/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
: *
dtype0
�
serving_default_input_5Placeholder*1
_output_shapes
:�����������*&
shape:�����������*
dtype0
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5conv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasz_mean_4/kernelz_mean_4/biasz_log_var_4/kernelz_log_var_4/biasdense_4/kerneldense_4/biasconv2d_transpose_16/kernelconv2d_transpose_16/biasconv2d_transpose_17/kernelconv2d_transpose_17/biasconv2d_transpose_18/kernelconv2d_transpose_18/biasconv2d_transpose_19/kernelconv2d_transpose_19/bias**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference_signature_wrapper_7657*1
_output_shapes
:�����������*"
Tin
2*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp#z_mean_4/kernel/Read/ReadVariableOp!z_mean_4/bias/Read/ReadVariableOp&z_log_var_4/kernel/Read/ReadVariableOp$z_log_var_4/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp.conv2d_transpose_16/kernel/Read/ReadVariableOp,conv2d_transpose_16/bias/Read/ReadVariableOp.conv2d_transpose_17/kernel/Read/ReadVariableOp,conv2d_transpose_17/bias/Read/ReadVariableOp.conv2d_transpose_18/kernel/Read/ReadVariableOp,conv2d_transpose_18/bias/Read/ReadVariableOp.conv2d_transpose_19/kernel/Read/ReadVariableOp,conv2d_transpose_19/bias/Read/ReadVariableOpAdam_4/iter/Read/ReadVariableOp!Adam_4/beta_1/Read/ReadVariableOp!Adam_4/beta_2/Read/ReadVariableOp Adam_4/decay/Read/ReadVariableOp(Adam_4/learning_rate/Read/ReadVariableOp-Adam_4/conv2d_16/kernel/m/Read/ReadVariableOp+Adam_4/conv2d_16/bias/m/Read/ReadVariableOp-Adam_4/conv2d_17/kernel/m/Read/ReadVariableOp+Adam_4/conv2d_17/bias/m/Read/ReadVariableOp-Adam_4/conv2d_18/kernel/m/Read/ReadVariableOp+Adam_4/conv2d_18/bias/m/Read/ReadVariableOp-Adam_4/conv2d_19/kernel/m/Read/ReadVariableOp+Adam_4/conv2d_19/bias/m/Read/ReadVariableOp,Adam_4/z_mean_4/kernel/m/Read/ReadVariableOp*Adam_4/z_mean_4/bias/m/Read/ReadVariableOp/Adam_4/z_log_var_4/kernel/m/Read/ReadVariableOp-Adam_4/z_log_var_4/bias/m/Read/ReadVariableOp+Adam_4/dense_4/kernel/m/Read/ReadVariableOp)Adam_4/dense_4/bias/m/Read/ReadVariableOp7Adam_4/conv2d_transpose_16/kernel/m/Read/ReadVariableOp5Adam_4/conv2d_transpose_16/bias/m/Read/ReadVariableOp7Adam_4/conv2d_transpose_17/kernel/m/Read/ReadVariableOp5Adam_4/conv2d_transpose_17/bias/m/Read/ReadVariableOp7Adam_4/conv2d_transpose_18/kernel/m/Read/ReadVariableOp5Adam_4/conv2d_transpose_18/bias/m/Read/ReadVariableOp7Adam_4/conv2d_transpose_19/kernel/m/Read/ReadVariableOp5Adam_4/conv2d_transpose_19/bias/m/Read/ReadVariableOp-Adam_4/conv2d_16/kernel/v/Read/ReadVariableOp+Adam_4/conv2d_16/bias/v/Read/ReadVariableOp-Adam_4/conv2d_17/kernel/v/Read/ReadVariableOp+Adam_4/conv2d_17/bias/v/Read/ReadVariableOp-Adam_4/conv2d_18/kernel/v/Read/ReadVariableOp+Adam_4/conv2d_18/bias/v/Read/ReadVariableOp-Adam_4/conv2d_19/kernel/v/Read/ReadVariableOp+Adam_4/conv2d_19/bias/v/Read/ReadVariableOp,Adam_4/z_mean_4/kernel/v/Read/ReadVariableOp*Adam_4/z_mean_4/bias/v/Read/ReadVariableOp/Adam_4/z_log_var_4/kernel/v/Read/ReadVariableOp-Adam_4/z_log_var_4/bias/v/Read/ReadVariableOp+Adam_4/dense_4/kernel/v/Read/ReadVariableOp)Adam_4/dense_4/bias/v/Read/ReadVariableOp7Adam_4/conv2d_transpose_16/kernel/v/Read/ReadVariableOp5Adam_4/conv2d_transpose_16/bias/v/Read/ReadVariableOp7Adam_4/conv2d_transpose_17/kernel/v/Read/ReadVariableOp5Adam_4/conv2d_transpose_17/bias/v/Read/ReadVariableOp7Adam_4/conv2d_transpose_18/kernel/v/Read/ReadVariableOp5Adam_4/conv2d_transpose_18/bias/v/Read/ReadVariableOp7Adam_4/conv2d_transpose_19/kernel/v/Read/ReadVariableOp5Adam_4/conv2d_transpose_19/bias/v/Read/ReadVariableOpConst*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_gradient_op_typePartitionedCall-7897*T
TinM
K2I	*&
f!R
__inference__traced_save_7896*
_output_shapes
: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasz_mean_4/kernelz_mean_4/biasz_log_var_4/kernelz_log_var_4/biasdense_4/kerneldense_4/biasconv2d_transpose_16/kernelconv2d_transpose_16/biasconv2d_transpose_17/kernelconv2d_transpose_17/biasconv2d_transpose_18/kernelconv2d_transpose_18/biasconv2d_transpose_19/kernelconv2d_transpose_19/biasAdam_4/iterAdam_4/beta_1Adam_4/beta_2Adam_4/decayAdam_4/learning_rateAdam_4/conv2d_16/kernel/mAdam_4/conv2d_16/bias/mAdam_4/conv2d_17/kernel/mAdam_4/conv2d_17/bias/mAdam_4/conv2d_18/kernel/mAdam_4/conv2d_18/bias/mAdam_4/conv2d_19/kernel/mAdam_4/conv2d_19/bias/mAdam_4/z_mean_4/kernel/mAdam_4/z_mean_4/bias/mAdam_4/z_log_var_4/kernel/mAdam_4/z_log_var_4/bias/mAdam_4/dense_4/kernel/mAdam_4/dense_4/bias/m#Adam_4/conv2d_transpose_16/kernel/m!Adam_4/conv2d_transpose_16/bias/m#Adam_4/conv2d_transpose_17/kernel/m!Adam_4/conv2d_transpose_17/bias/m#Adam_4/conv2d_transpose_18/kernel/m!Adam_4/conv2d_transpose_18/bias/m#Adam_4/conv2d_transpose_19/kernel/m!Adam_4/conv2d_transpose_19/bias/mAdam_4/conv2d_16/kernel/vAdam_4/conv2d_16/bias/vAdam_4/conv2d_17/kernel/vAdam_4/conv2d_17/bias/vAdam_4/conv2d_18/kernel/vAdam_4/conv2d_18/bias/vAdam_4/conv2d_19/kernel/vAdam_4/conv2d_19/bias/vAdam_4/z_mean_4/kernel/vAdam_4/z_mean_4/bias/vAdam_4/z_log_var_4/kernel/vAdam_4/z_log_var_4/bias/vAdam_4/dense_4/kernel/vAdam_4/dense_4/bias/v#Adam_4/conv2d_transpose_16/kernel/v!Adam_4/conv2d_transpose_16/bias/v#Adam_4/conv2d_transpose_17/kernel/v!Adam_4/conv2d_transpose_17/bias/v#Adam_4/conv2d_transpose_18/kernel/v!Adam_4/conv2d_transpose_18/bias/v#Adam_4/conv2d_transpose_19/kernel/v!Adam_4/conv2d_transpose_19/bias/v*
Tout
2*S
TinL
J2H*
_output_shapes
: *)
f$R"
 __inference__traced_restore_8122*+
_gradient_op_typePartitionedCall-8123**
config_proto

GPU 

CPU2J 8��
ۃ
� 
__inference__traced_save_7896
file_prefix/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop.
*savev2_z_mean_4_kernel_read_readvariableop,
(savev2_z_mean_4_bias_read_readvariableop1
-savev2_z_log_var_4_kernel_read_readvariableop/
+savev2_z_log_var_4_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop9
5savev2_conv2d_transpose_16_kernel_read_readvariableop7
3savev2_conv2d_transpose_16_bias_read_readvariableop9
5savev2_conv2d_transpose_17_kernel_read_readvariableop7
3savev2_conv2d_transpose_17_bias_read_readvariableop9
5savev2_conv2d_transpose_18_kernel_read_readvariableop7
3savev2_conv2d_transpose_18_bias_read_readvariableop9
5savev2_conv2d_transpose_19_kernel_read_readvariableop7
3savev2_conv2d_transpose_19_bias_read_readvariableop*
&savev2_adam_4_iter_read_readvariableop	,
(savev2_adam_4_beta_1_read_readvariableop,
(savev2_adam_4_beta_2_read_readvariableop+
'savev2_adam_4_decay_read_readvariableop3
/savev2_adam_4_learning_rate_read_readvariableop8
4savev2_adam_4_conv2d_16_kernel_m_read_readvariableop6
2savev2_adam_4_conv2d_16_bias_m_read_readvariableop8
4savev2_adam_4_conv2d_17_kernel_m_read_readvariableop6
2savev2_adam_4_conv2d_17_bias_m_read_readvariableop8
4savev2_adam_4_conv2d_18_kernel_m_read_readvariableop6
2savev2_adam_4_conv2d_18_bias_m_read_readvariableop8
4savev2_adam_4_conv2d_19_kernel_m_read_readvariableop6
2savev2_adam_4_conv2d_19_bias_m_read_readvariableop7
3savev2_adam_4_z_mean_4_kernel_m_read_readvariableop5
1savev2_adam_4_z_mean_4_bias_m_read_readvariableop:
6savev2_adam_4_z_log_var_4_kernel_m_read_readvariableop8
4savev2_adam_4_z_log_var_4_bias_m_read_readvariableop6
2savev2_adam_4_dense_4_kernel_m_read_readvariableop4
0savev2_adam_4_dense_4_bias_m_read_readvariableopB
>savev2_adam_4_conv2d_transpose_16_kernel_m_read_readvariableop@
<savev2_adam_4_conv2d_transpose_16_bias_m_read_readvariableopB
>savev2_adam_4_conv2d_transpose_17_kernel_m_read_readvariableop@
<savev2_adam_4_conv2d_transpose_17_bias_m_read_readvariableopB
>savev2_adam_4_conv2d_transpose_18_kernel_m_read_readvariableop@
<savev2_adam_4_conv2d_transpose_18_bias_m_read_readvariableopB
>savev2_adam_4_conv2d_transpose_19_kernel_m_read_readvariableop@
<savev2_adam_4_conv2d_transpose_19_bias_m_read_readvariableop8
4savev2_adam_4_conv2d_16_kernel_v_read_readvariableop6
2savev2_adam_4_conv2d_16_bias_v_read_readvariableop8
4savev2_adam_4_conv2d_17_kernel_v_read_readvariableop6
2savev2_adam_4_conv2d_17_bias_v_read_readvariableop8
4savev2_adam_4_conv2d_18_kernel_v_read_readvariableop6
2savev2_adam_4_conv2d_18_bias_v_read_readvariableop8
4savev2_adam_4_conv2d_19_kernel_v_read_readvariableop6
2savev2_adam_4_conv2d_19_bias_v_read_readvariableop7
3savev2_adam_4_z_mean_4_kernel_v_read_readvariableop5
1savev2_adam_4_z_mean_4_bias_v_read_readvariableop:
6savev2_adam_4_z_log_var_4_kernel_v_read_readvariableop8
4savev2_adam_4_z_log_var_4_bias_v_read_readvariableop6
2savev2_adam_4_dense_4_kernel_v_read_readvariableop4
0savev2_adam_4_dense_4_bias_v_read_readvariableopB
>savev2_adam_4_conv2d_transpose_16_kernel_v_read_readvariableop@
<savev2_adam_4_conv2d_transpose_16_bias_v_read_readvariableopB
>savev2_adam_4_conv2d_transpose_17_kernel_v_read_readvariableop@
<savev2_adam_4_conv2d_transpose_17_bias_v_read_readvariableopB
>savev2_adam_4_conv2d_transpose_18_kernel_v_read_readvariableop@
<savev2_adam_4_conv2d_transpose_18_bias_v_read_readvariableopB
>savev2_adam_4_conv2d_transpose_19_kernel_v_read_readvariableop@
<savev2_adam_4_conv2d_transpose_19_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2_1�SaveV2�
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_3435fd00f03241209e9e624ec6bda8d8/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
value	B :*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �(
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*�'
value�'B�'GB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0�
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:G*�
value�B�GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop*savev2_z_mean_4_kernel_read_readvariableop(savev2_z_mean_4_bias_read_readvariableop-savev2_z_log_var_4_kernel_read_readvariableop+savev2_z_log_var_4_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop5savev2_conv2d_transpose_16_kernel_read_readvariableop3savev2_conv2d_transpose_16_bias_read_readvariableop5savev2_conv2d_transpose_17_kernel_read_readvariableop3savev2_conv2d_transpose_17_bias_read_readvariableop5savev2_conv2d_transpose_18_kernel_read_readvariableop3savev2_conv2d_transpose_18_bias_read_readvariableop5savev2_conv2d_transpose_19_kernel_read_readvariableop3savev2_conv2d_transpose_19_bias_read_readvariableop&savev2_adam_4_iter_read_readvariableop(savev2_adam_4_beta_1_read_readvariableop(savev2_adam_4_beta_2_read_readvariableop'savev2_adam_4_decay_read_readvariableop/savev2_adam_4_learning_rate_read_readvariableop4savev2_adam_4_conv2d_16_kernel_m_read_readvariableop2savev2_adam_4_conv2d_16_bias_m_read_readvariableop4savev2_adam_4_conv2d_17_kernel_m_read_readvariableop2savev2_adam_4_conv2d_17_bias_m_read_readvariableop4savev2_adam_4_conv2d_18_kernel_m_read_readvariableop2savev2_adam_4_conv2d_18_bias_m_read_readvariableop4savev2_adam_4_conv2d_19_kernel_m_read_readvariableop2savev2_adam_4_conv2d_19_bias_m_read_readvariableop3savev2_adam_4_z_mean_4_kernel_m_read_readvariableop1savev2_adam_4_z_mean_4_bias_m_read_readvariableop6savev2_adam_4_z_log_var_4_kernel_m_read_readvariableop4savev2_adam_4_z_log_var_4_bias_m_read_readvariableop2savev2_adam_4_dense_4_kernel_m_read_readvariableop0savev2_adam_4_dense_4_bias_m_read_readvariableop>savev2_adam_4_conv2d_transpose_16_kernel_m_read_readvariableop<savev2_adam_4_conv2d_transpose_16_bias_m_read_readvariableop>savev2_adam_4_conv2d_transpose_17_kernel_m_read_readvariableop<savev2_adam_4_conv2d_transpose_17_bias_m_read_readvariableop>savev2_adam_4_conv2d_transpose_18_kernel_m_read_readvariableop<savev2_adam_4_conv2d_transpose_18_bias_m_read_readvariableop>savev2_adam_4_conv2d_transpose_19_kernel_m_read_readvariableop<savev2_adam_4_conv2d_transpose_19_bias_m_read_readvariableop4savev2_adam_4_conv2d_16_kernel_v_read_readvariableop2savev2_adam_4_conv2d_16_bias_v_read_readvariableop4savev2_adam_4_conv2d_17_kernel_v_read_readvariableop2savev2_adam_4_conv2d_17_bias_v_read_readvariableop4savev2_adam_4_conv2d_18_kernel_v_read_readvariableop2savev2_adam_4_conv2d_18_bias_v_read_readvariableop4savev2_adam_4_conv2d_19_kernel_v_read_readvariableop2savev2_adam_4_conv2d_19_bias_v_read_readvariableop3savev2_adam_4_z_mean_4_kernel_v_read_readvariableop1savev2_adam_4_z_mean_4_bias_v_read_readvariableop6savev2_adam_4_z_log_var_4_kernel_v_read_readvariableop4savev2_adam_4_z_log_var_4_bias_v_read_readvariableop2savev2_adam_4_dense_4_kernel_v_read_readvariableop0savev2_adam_4_dense_4_bias_v_read_readvariableop>savev2_adam_4_conv2d_transpose_16_kernel_v_read_readvariableop<savev2_adam_4_conv2d_transpose_16_bias_v_read_readvariableop>savev2_adam_4_conv2d_transpose_17_kernel_v_read_readvariableop<savev2_adam_4_conv2d_transpose_17_bias_v_read_readvariableop>savev2_adam_4_conv2d_transpose_18_kernel_v_read_readvariableop<savev2_adam_4_conv2d_transpose_18_bias_v_read_readvariableop>savev2_adam_4_conv2d_transpose_19_kernel_v_read_readvariableop<savev2_adam_4_conv2d_transpose_19_bias_v_read_readvariableop"/device:CPU:0*U
dtypesK
I2G	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
_output_shapes
:*
N�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints	^SaveV2_1^SaveV2*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : :  : : @:@:@@:@:
��:�:
��:�:
��:�:@@:@: @: :  : : :: : : : : : : :  : : @:@:@@:@:
��:�:
��:�:
��:�:@@:@: @: :  : : :: : :  : : @:@:@@:@:
��:�:
��:�:
��:�:@@:@: @: :  : : :: 2
SaveV2_1SaveV2_12(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G :H 
�
�	
"__inference_signature_wrapper_7657
input_5,
(statefulpartitionedcall_conv2d_16_kernel*
&statefulpartitionedcall_conv2d_16_bias,
(statefulpartitionedcall_conv2d_17_kernel*
&statefulpartitionedcall_conv2d_17_bias,
(statefulpartitionedcall_conv2d_18_kernel*
&statefulpartitionedcall_conv2d_18_bias,
(statefulpartitionedcall_conv2d_19_kernel*
&statefulpartitionedcall_conv2d_19_bias+
'statefulpartitionedcall_z_mean_4_kernel)
%statefulpartitionedcall_z_mean_4_bias.
*statefulpartitionedcall_z_log_var_4_kernel,
(statefulpartitionedcall_z_log_var_4_bias*
&statefulpartitionedcall_dense_4_kernel(
$statefulpartitionedcall_dense_4_bias6
2statefulpartitionedcall_conv2d_transpose_16_kernel4
0statefulpartitionedcall_conv2d_transpose_16_bias6
2statefulpartitionedcall_conv2d_transpose_17_kernel4
0statefulpartitionedcall_conv2d_transpose_17_bias6
2statefulpartitionedcall_conv2d_transpose_18_kernel4
0statefulpartitionedcall_conv2d_transpose_18_bias6
2statefulpartitionedcall_conv2d_transpose_19_kernel4
0statefulpartitionedcall_conv2d_transpose_19_bias
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinput_5(statefulpartitionedcall_conv2d_16_kernel&statefulpartitionedcall_conv2d_16_bias(statefulpartitionedcall_conv2d_17_kernel&statefulpartitionedcall_conv2d_17_bias(statefulpartitionedcall_conv2d_18_kernel&statefulpartitionedcall_conv2d_18_bias(statefulpartitionedcall_conv2d_19_kernel&statefulpartitionedcall_conv2d_19_bias'statefulpartitionedcall_z_mean_4_kernel%statefulpartitionedcall_z_mean_4_bias*statefulpartitionedcall_z_log_var_4_kernel(statefulpartitionedcall_z_log_var_4_bias&statefulpartitionedcall_dense_4_kernel$statefulpartitionedcall_dense_4_bias2statefulpartitionedcall_conv2d_transpose_16_kernel0statefulpartitionedcall_conv2d_transpose_16_bias2statefulpartitionedcall_conv2d_transpose_17_kernel0statefulpartitionedcall_conv2d_transpose_17_bias2statefulpartitionedcall_conv2d_transpose_18_kernel0statefulpartitionedcall_conv2d_transpose_18_bias2statefulpartitionedcall_conv2d_transpose_19_kernel0statefulpartitionedcall_conv2d_transpose_19_bias*"
Tin
2**
config_proto

GPU 

CPU2J 8*+
_gradient_op_typePartitionedCall-7632*1
_output_shapes
:�����������*
Tout
2*(
f#R!
__inference__wrapped_model_7626�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*1
_output_shapes
:�����������*
T0"
identityIdentity:output:0*�
_input_shapesw
u:�����������::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :	 :
 : : : : : : : : : : : : :' #
!
_user_specified_name	input_5: : : : 
��
�
__inference__wrapped_model_7626
input_5<
8model_3_conv2d_16_conv2d_readvariableop_conv2d_16_kernel;
7model_3_conv2d_16_biasadd_readvariableop_conv2d_16_bias<
8model_3_conv2d_17_conv2d_readvariableop_conv2d_17_kernel;
7model_3_conv2d_17_biasadd_readvariableop_conv2d_17_bias<
8model_3_conv2d_18_conv2d_readvariableop_conv2d_18_kernel;
7model_3_conv2d_18_biasadd_readvariableop_conv2d_18_bias<
8model_3_conv2d_19_conv2d_readvariableop_conv2d_19_kernel;
7model_3_conv2d_19_biasadd_readvariableop_conv2d_19_bias8
4model_3_z_mean_matmul_readvariableop_z_mean_4_kernel7
3model_3_z_mean_biasadd_readvariableop_z_mean_4_bias>
:model_3_z_log_var_matmul_readvariableop_z_log_var_4_kernel=
9model_3_z_log_var_biasadd_readvariableop_z_log_var_4_bias8
4model_3_dense_4_matmul_readvariableop_dense_4_kernel7
3model_3_dense_4_biasadd_readvariableop_dense_4_biasZ
Vmodel_3_conv2d_transpose_16_conv2d_transpose_readvariableop_conv2d_transpose_16_kernelO
Kmodel_3_conv2d_transpose_16_biasadd_readvariableop_conv2d_transpose_16_biasZ
Vmodel_3_conv2d_transpose_17_conv2d_transpose_readvariableop_conv2d_transpose_17_kernelO
Kmodel_3_conv2d_transpose_17_biasadd_readvariableop_conv2d_transpose_17_biasZ
Vmodel_3_conv2d_transpose_18_conv2d_transpose_readvariableop_conv2d_transpose_18_kernelO
Kmodel_3_conv2d_transpose_18_biasadd_readvariableop_conv2d_transpose_18_biasZ
Vmodel_3_conv2d_transpose_19_conv2d_transpose_readvariableop_conv2d_transpose_19_kernelO
Kmodel_3_conv2d_transpose_19_biasadd_readvariableop_conv2d_transpose_19_bias
identity��2model_3/conv2d_transpose_16/BiasAdd/ReadVariableOp�$model_3/z_mean/MatMul/ReadVariableOp�(model_3/conv2d_16/BiasAdd/ReadVariableOp�(model_3/conv2d_18/BiasAdd/ReadVariableOp�;model_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOp�&model_3/dense_4/BiasAdd/ReadVariableOp�'model_3/conv2d_19/Conv2D/ReadVariableOp�'model_3/z_log_var/MatMul/ReadVariableOp�;model_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOp�'model_3/conv2d_16/Conv2D/ReadVariableOp�%model_3/dense_4/MatMul/ReadVariableOp�(model_3/conv2d_17/BiasAdd/ReadVariableOp�;model_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOp�2model_3/conv2d_transpose_18/BiasAdd/ReadVariableOp�%model_3/z_mean/BiasAdd/ReadVariableOp�2model_3/conv2d_transpose_17/BiasAdd/ReadVariableOp�'model_3/conv2d_18/Conv2D/ReadVariableOp�;model_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOp�'model_3/conv2d_17/Conv2D/ReadVariableOp�2model_3/conv2d_transpose_19/BiasAdd/ReadVariableOp�(model_3/conv2d_19/BiasAdd/ReadVariableOp�(model_3/z_log_var/BiasAdd/ReadVariableOp�
'model_3/conv2d_16/Conv2D/ReadVariableOpReadVariableOp8model_3_conv2d_16_conv2d_readvariableop_conv2d_16_kernel*&
_output_shapes
: *
dtype0�
model_3/conv2d_16/Conv2DConv2Dinput_5/model_3/conv2d_16/Conv2D/ReadVariableOp:value:0*
strides
*0
_output_shapes
:���������x� *
paddingSAME*
T0�
(model_3/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp7model_3_conv2d_16_biasadd_readvariableop_conv2d_16_bias*
dtype0*
_output_shapes
: �
model_3/conv2d_16/BiasAddBiasAdd!model_3/conv2d_16/Conv2D:output:00model_3/conv2d_16/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:���������x� *
T0}
model_3/conv2d_16/ReluRelu"model_3/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:���������x� �
'model_3/conv2d_17/Conv2D/ReadVariableOpReadVariableOp8model_3_conv2d_17_conv2d_readvariableop_conv2d_17_kernel*
dtype0*&
_output_shapes
:  �
model_3/conv2d_17/Conv2DConv2D$model_3/conv2d_16/Relu:activations:0/model_3/conv2d_17/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:���������<P *
paddingSAME*
strides
*
T0�
(model_3/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp7model_3_conv2d_17_biasadd_readvariableop_conv2d_17_bias*
dtype0*
_output_shapes
: �
model_3/conv2d_17/BiasAddBiasAdd!model_3/conv2d_17/Conv2D:output:00model_3/conv2d_17/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:���������<P *
T0|
model_3/conv2d_17/ReluRelu"model_3/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:���������<P �
'model_3/conv2d_18/Conv2D/ReadVariableOpReadVariableOp8model_3_conv2d_18_conv2d_readvariableop_conv2d_18_kernel*&
_output_shapes
: @*
dtype0�
model_3/conv2d_18/Conv2DConv2D$model_3/conv2d_17/Relu:activations:0/model_3/conv2d_18/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:���������(@*
T0*
strides
*
paddingSAME�
(model_3/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp7model_3_conv2d_18_biasadd_readvariableop_conv2d_18_bias*
dtype0*
_output_shapes
:@�
model_3/conv2d_18/BiasAddBiasAdd!model_3/conv2d_18/Conv2D:output:00model_3/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������(@|
model_3/conv2d_18/ReluRelu"model_3/conv2d_18/BiasAdd:output:0*/
_output_shapes
:���������(@*
T0�
'model_3/conv2d_19/Conv2D/ReadVariableOpReadVariableOp8model_3_conv2d_19_conv2d_readvariableop_conv2d_19_kernel*
dtype0*&
_output_shapes
:@@�
model_3/conv2d_19/Conv2DConv2D$model_3/conv2d_18/Relu:activations:0/model_3/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*
paddingSAME*
strides
*/
_output_shapes
:���������@�
(model_3/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp7model_3_conv2d_19_biasadd_readvariableop_conv2d_19_bias*
_output_shapes
:@*
dtype0�
model_3/conv2d_19/BiasAddBiasAdd!model_3/conv2d_19/Conv2D:output:00model_3/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@|
model_3/conv2d_19/ReluRelu"model_3/conv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:���������@k
model_3/flatten_4/ShapeShape$model_3/conv2d_19/Relu:activations:0*
T0*
_output_shapes
:o
%model_3/flatten_4/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: q
'model_3/flatten_4/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:q
'model_3/flatten_4/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
model_3/flatten_4/strided_sliceStridedSlice model_3/flatten_4/Shape:output:0.model_3/flatten_4/strided_slice/stack:output:00model_3/flatten_4/strided_slice/stack_1:output:00model_3/flatten_4/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskl
!model_3/flatten_4/Reshape/shape/1Const*
_output_shapes
: *
valueB :
���������*
dtype0�
model_3/flatten_4/Reshape/shapePack(model_3/flatten_4/strided_slice:output:0*model_3/flatten_4/Reshape/shape/1:output:0*
T0*
_output_shapes
:*
N�
model_3/flatten_4/ReshapeReshape$model_3/conv2d_19/Relu:activations:0(model_3/flatten_4/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
$model_3/z_mean/MatMul/ReadVariableOpReadVariableOp4model_3_z_mean_matmul_readvariableop_z_mean_4_kernel*
dtype0* 
_output_shapes
:
���
model_3/z_mean/MatMulMatMul"model_3/flatten_4/Reshape:output:0,model_3/z_mean/MatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
%model_3/z_mean/BiasAdd/ReadVariableOpReadVariableOp3model_3_z_mean_biasadd_readvariableop_z_mean_4_bias*
dtype0*
_output_shapes	
:��
model_3/z_mean/BiasAddBiasAddmodel_3/z_mean/MatMul:product:0-model_3/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'model_3/z_log_var/MatMul/ReadVariableOpReadVariableOp:model_3_z_log_var_matmul_readvariableop_z_log_var_4_kernel* 
_output_shapes
:
��*
dtype0�
model_3/z_log_var/MatMulMatMul"model_3/flatten_4/Reshape:output:0/model_3/z_log_var/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(model_3/z_log_var/BiasAdd/ReadVariableOpReadVariableOp9model_3_z_log_var_biasadd_readvariableop_z_log_var_4_bias*
_output_shapes	
:�*
dtype0�
model_3/z_log_var/BiasAddBiasAdd"model_3/z_log_var/MatMul:product:00model_3/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
model_3/sampling_4/ShapeShapemodel_3/z_mean/BiasAdd:output:0*
_output_shapes
:*
T0p
&model_3/sampling_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(model_3/sampling_4/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:r
(model_3/sampling_4/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
 model_3/sampling_4/strided_sliceStridedSlice!model_3/sampling_4/Shape:output:0/model_3/sampling_4/strided_slice/stack:output:01model_3/sampling_4/strided_slice/stack_1:output:01model_3/sampling_4/strided_slice/stack_2:output:0*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0i
model_3/sampling_4/Shape_1Shapemodel_3/z_mean/BiasAdd:output:0*
_output_shapes
:*
T0r
(model_3/sampling_4/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0t
*model_3/sampling_4/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:t
*model_3/sampling_4/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
"model_3/sampling_4/strided_slice_1StridedSlice#model_3/sampling_4/Shape_1:output:01model_3/sampling_4/strided_slice_1/stack:output:03model_3/sampling_4/strided_slice_1/stack_1:output:03model_3/sampling_4/strided_slice_1/stack_2:output:0*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: �
&model_3/sampling_4/random_normal/shapePack)model_3/sampling_4/strided_slice:output:0+model_3/sampling_4/strided_slice_1:output:0*
_output_shapes
:*
N*
T0j
%model_3/sampling_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    l
'model_3/sampling_4/random_normal/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0�
5model_3/sampling_4/random_normal/RandomStandardNormalRandomStandardNormal/model_3/sampling_4/random_normal/shape:output:0*0
_output_shapes
:������������������*
T0*
dtype0*
seed2���*
seed���)�
$model_3/sampling_4/random_normal/mulMul>model_3/sampling_4/random_normal/RandomStandardNormal:output:00model_3/sampling_4/random_normal/stddev:output:0*
T0*0
_output_shapes
:�������������������
 model_3/sampling_4/random_normalAdd(model_3/sampling_4/random_normal/mul:z:0.model_3/sampling_4/random_normal/mean:output:0*0
_output_shapes
:������������������*
T0]
model_3/sampling_4/mul/xConst*
dtype0*
valueB
 *   ?*
_output_shapes
: �
model_3/sampling_4/mulMul!model_3/sampling_4/mul/x:output:0"model_3/z_log_var/BiasAdd:output:0*(
_output_shapes
:����������*
T0l
model_3/sampling_4/ExpExpmodel_3/sampling_4/mul:z:0*(
_output_shapes
:����������*
T0�
model_3/sampling_4/mul_1Mulmodel_3/sampling_4/Exp:y:0$model_3/sampling_4/random_normal:z:0*(
_output_shapes
:����������*
T0�
model_3/sampling_4/addAddmodel_3/z_mean/BiasAdd:output:0model_3/sampling_4/mul_1:z:0*(
_output_shapes
:����������*
T0�
%model_3/dense_4/MatMul/ReadVariableOpReadVariableOp4model_3_dense_4_matmul_readvariableop_dense_4_kernel*
dtype0* 
_output_shapes
:
���
model_3/dense_4/MatMulMatMulmodel_3/sampling_4/add:z:0-model_3/dense_4/MatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
&model_3/dense_4/BiasAdd/ReadVariableOpReadVariableOp3model_3_dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes	
:�*
dtype0�
model_3/dense_4/BiasAddBiasAdd model_3/dense_4/MatMul:product:0.model_3/dense_4/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0q
model_3/dense_4/ReluRelu model_3/dense_4/BiasAdd:output:0*(
_output_shapes
:����������*
T0i
model_3/reshape_4/ShapeShape"model_3/dense_4/Relu:activations:0*
T0*
_output_shapes
:o
%model_3/reshape_4/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0q
'model_3/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_3/reshape_4/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0�
model_3/reshape_4/strided_sliceStridedSlice model_3/reshape_4/Shape:output:0.model_3/reshape_4/strided_slice/stack:output:00model_3/reshape_4/strided_slice/stack_1:output:00model_3/reshape_4/strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: c
!model_3/reshape_4/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: c
!model_3/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model_3/reshape_4/Reshape/shape/3Const*
value	B :@*
dtype0*
_output_shapes
: �
model_3/reshape_4/Reshape/shapePack(model_3/reshape_4/strided_slice:output:0*model_3/reshape_4/Reshape/shape/1:output:0*model_3/reshape_4/Reshape/shape/2:output:0*model_3/reshape_4/Reshape/shape/3:output:0*
N*
_output_shapes
:*
T0�
model_3/reshape_4/ReshapeReshape"model_3/dense_4/Relu:activations:0(model_3/reshape_4/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@s
!model_3/conv2d_transpose_16/ShapeShape"model_3/reshape_4/Reshape:output:0*
_output_shapes
:*
T0y
/model_3/conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0{
1model_3/conv2d_transpose_16/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:{
1model_3/conv2d_transpose_16/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0�
)model_3/conv2d_transpose_16/strided_sliceStridedSlice*model_3/conv2d_transpose_16/Shape:output:08model_3/conv2d_transpose_16/strided_slice/stack:output:0:model_3/conv2d_transpose_16/strided_slice/stack_1:output:0:model_3/conv2d_transpose_16/strided_slice/stack_2:output:0*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0{
1model_3/conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
valueB:*
dtype0}
3model_3/conv2d_transpose_16/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:}
3model_3/conv2d_transpose_16/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
+model_3/conv2d_transpose_16/strided_slice_1StridedSlice*model_3/conv2d_transpose_16/Shape:output:0:model_3/conv2d_transpose_16/strided_slice_1/stack:output:0<model_3/conv2d_transpose_16/strided_slice_1/stack_1:output:0<model_3/conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
shrink_axis_mask*
_output_shapes
: *
T0{
1model_3/conv2d_transpose_16/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:}
3model_3/conv2d_transpose_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_3/conv2d_transpose_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+model_3/conv2d_transpose_16/strided_slice_2StridedSlice*model_3/conv2d_transpose_16/Shape:output:0:model_3/conv2d_transpose_16/strided_slice_2/stack:output:0<model_3/conv2d_transpose_16/strided_slice_2/stack_1:output:0<model_3/conv2d_transpose_16/strided_slice_2/stack_2:output:0*
shrink_axis_mask*
T0*
_output_shapes
: *
Index0c
!model_3/conv2d_transpose_16/mul/yConst*
value	B :*
_output_shapes
: *
dtype0�
model_3/conv2d_transpose_16/mulMul4model_3/conv2d_transpose_16/strided_slice_1:output:0*model_3/conv2d_transpose_16/mul/y:output:0*
T0*
_output_shapes
: e
#model_3/conv2d_transpose_16/mul_1/yConst*
dtype0*
_output_shapes
: *
value	B :�
!model_3/conv2d_transpose_16/mul_1Mul4model_3/conv2d_transpose_16/strided_slice_2:output:0,model_3/conv2d_transpose_16/mul_1/y:output:0*
T0*
_output_shapes
: e
#model_3/conv2d_transpose_16/stack/3Const*
dtype0*
_output_shapes
: *
value	B :@�
!model_3/conv2d_transpose_16/stackPack2model_3/conv2d_transpose_16/strided_slice:output:0#model_3/conv2d_transpose_16/mul:z:0%model_3/conv2d_transpose_16/mul_1:z:0,model_3/conv2d_transpose_16/stack/3:output:0*
T0*
N*
_output_shapes
:{
1model_3/conv2d_transpose_16/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:}
3model_3/conv2d_transpose_16/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:}
3model_3/conv2d_transpose_16/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:�
+model_3/conv2d_transpose_16/strided_slice_3StridedSlice*model_3/conv2d_transpose_16/stack:output:0:model_3/conv2d_transpose_16/strided_slice_3/stack:output:0<model_3/conv2d_transpose_16/strided_slice_3/stack_1:output:0<model_3/conv2d_transpose_16/strided_slice_3/stack_2:output:0*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0�
;model_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOpVmodel_3_conv2d_transpose_16_conv2d_transpose_readvariableop_conv2d_transpose_16_kernel*&
_output_shapes
:@@*
dtype0�
,model_3/conv2d_transpose_16/conv2d_transposeConv2DBackpropInput*model_3/conv2d_transpose_16/stack:output:0Cmodel_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0"model_3/reshape_4/Reshape:output:0*/
_output_shapes
:���������(@*
paddingSAME*
strides
*
T0�
2model_3/conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOpKmodel_3_conv2d_transpose_16_biasadd_readvariableop_conv2d_transpose_16_bias*
_output_shapes
:@*
dtype0�
#model_3/conv2d_transpose_16/BiasAddBiasAdd5model_3/conv2d_transpose_16/conv2d_transpose:output:0:model_3/conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:���������(@*
T0�
 model_3/conv2d_transpose_16/ReluRelu,model_3/conv2d_transpose_16/BiasAdd:output:0*/
_output_shapes
:���������(@*
T0
!model_3/conv2d_transpose_17/ShapeShape.model_3/conv2d_transpose_16/Relu:activations:0*
_output_shapes
:*
T0y
/model_3/conv2d_transpose_17/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0{
1model_3/conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_3/conv2d_transpose_17/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0�
)model_3/conv2d_transpose_17/strided_sliceStridedSlice*model_3/conv2d_transpose_17/Shape:output:08model_3/conv2d_transpose_17/strided_slice/stack:output:0:model_3/conv2d_transpose_17/strided_slice/stack_1:output:0:model_3/conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model_3/conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
valueB:*
dtype0}
3model_3/conv2d_transpose_17/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:}
3model_3/conv2d_transpose_17/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:�
+model_3/conv2d_transpose_17/strided_slice_1StridedSlice*model_3/conv2d_transpose_17/Shape:output:0:model_3/conv2d_transpose_17/strided_slice_1/stack:output:0<model_3/conv2d_transpose_17/strided_slice_1/stack_1:output:0<model_3/conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
_output_shapes
: *
T0*
shrink_axis_mask{
1model_3/conv2d_transpose_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model_3/conv2d_transpose_17/strided_slice_2/stack_1Const*
valueB:*
_output_shapes
:*
dtype0}
3model_3/conv2d_transpose_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+model_3/conv2d_transpose_17/strided_slice_2StridedSlice*model_3/conv2d_transpose_17/Shape:output:0:model_3/conv2d_transpose_17/strided_slice_2/stack:output:0<model_3/conv2d_transpose_17/strided_slice_2/stack_1:output:0<model_3/conv2d_transpose_17/strided_slice_2/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskc
!model_3/conv2d_transpose_17/mul/yConst*
dtype0*
value	B :*
_output_shapes
: �
model_3/conv2d_transpose_17/mulMul4model_3/conv2d_transpose_17/strided_slice_1:output:0*model_3/conv2d_transpose_17/mul/y:output:0*
_output_shapes
: *
T0e
#model_3/conv2d_transpose_17/mul_1/yConst*
value	B :*
dtype0*
_output_shapes
: �
!model_3/conv2d_transpose_17/mul_1Mul4model_3/conv2d_transpose_17/strided_slice_2:output:0,model_3/conv2d_transpose_17/mul_1/y:output:0*
_output_shapes
: *
T0e
#model_3/conv2d_transpose_17/stack/3Const*
value	B : *
dtype0*
_output_shapes
: �
!model_3/conv2d_transpose_17/stackPack2model_3/conv2d_transpose_17/strided_slice:output:0#model_3/conv2d_transpose_17/mul:z:0%model_3/conv2d_transpose_17/mul_1:z:0,model_3/conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_3/conv2d_transpose_17/strided_slice_3/stackConst*
_output_shapes
:*
valueB: *
dtype0}
3model_3/conv2d_transpose_17/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB:}
3model_3/conv2d_transpose_17/strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
+model_3/conv2d_transpose_17/strided_slice_3StridedSlice*model_3/conv2d_transpose_17/stack:output:0:model_3/conv2d_transpose_17/strided_slice_3/stack:output:0<model_3/conv2d_transpose_17/strided_slice_3/stack_1:output:0<model_3/conv2d_transpose_17/strided_slice_3/stack_2:output:0*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0�
;model_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOpVmodel_3_conv2d_transpose_17_conv2d_transpose_readvariableop_conv2d_transpose_17_kernel*&
_output_shapes
: @*
dtype0�
,model_3/conv2d_transpose_17/conv2d_transposeConv2DBackpropInput*model_3/conv2d_transpose_17/stack:output:0Cmodel_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0.model_3/conv2d_transpose_16/Relu:activations:0*
paddingSAME*
strides
*
T0*/
_output_shapes
:���������<P �
2model_3/conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOpKmodel_3_conv2d_transpose_17_biasadd_readvariableop_conv2d_transpose_17_bias*
dtype0*
_output_shapes
: �
#model_3/conv2d_transpose_17/BiasAddBiasAdd5model_3/conv2d_transpose_17/conv2d_transpose:output:0:model_3/conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<P �
 model_3/conv2d_transpose_17/ReluRelu,model_3/conv2d_transpose_17/BiasAdd:output:0*
T0*/
_output_shapes
:���������<P 
!model_3/conv2d_transpose_18/ShapeShape.model_3/conv2d_transpose_17/Relu:activations:0*
T0*
_output_shapes
:y
/model_3/conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_3/conv2d_transpose_18/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:{
1model_3/conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
)model_3/conv2d_transpose_18/strided_sliceStridedSlice*model_3/conv2d_transpose_18/Shape:output:08model_3/conv2d_transpose_18/strided_slice/stack:output:0:model_3/conv2d_transpose_18/strided_slice/stack_1:output:0:model_3/conv2d_transpose_18/strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
_output_shapes
: *
Index0{
1model_3/conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model_3/conv2d_transpose_18/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:}
3model_3/conv2d_transpose_18/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
+model_3/conv2d_transpose_18/strided_slice_1StridedSlice*model_3/conv2d_transpose_18/Shape:output:0:model_3/conv2d_transpose_18/strided_slice_1/stack:output:0<model_3/conv2d_transpose_18/strided_slice_1/stack_1:output:0<model_3/conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model_3/conv2d_transpose_18/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:}
3model_3/conv2d_transpose_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_3/conv2d_transpose_18/strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0�
+model_3/conv2d_transpose_18/strided_slice_2StridedSlice*model_3/conv2d_transpose_18/Shape:output:0:model_3/conv2d_transpose_18/strided_slice_2/stack:output:0<model_3/conv2d_transpose_18/strided_slice_2/stack_1:output:0<model_3/conv2d_transpose_18/strided_slice_2/stack_2:output:0*
Index0*
_output_shapes
: *
shrink_axis_mask*
T0c
!model_3/conv2d_transpose_18/mul/yConst*
dtype0*
_output_shapes
: *
value	B :�
model_3/conv2d_transpose_18/mulMul4model_3/conv2d_transpose_18/strided_slice_1:output:0*model_3/conv2d_transpose_18/mul/y:output:0*
T0*
_output_shapes
: e
#model_3/conv2d_transpose_18/mul_1/yConst*
value	B :*
_output_shapes
: *
dtype0�
!model_3/conv2d_transpose_18/mul_1Mul4model_3/conv2d_transpose_18/strided_slice_2:output:0,model_3/conv2d_transpose_18/mul_1/y:output:0*
T0*
_output_shapes
: e
#model_3/conv2d_transpose_18/stack/3Const*
dtype0*
_output_shapes
: *
value	B : �
!model_3/conv2d_transpose_18/stackPack2model_3/conv2d_transpose_18/strided_slice:output:0#model_3/conv2d_transpose_18/mul:z:0%model_3/conv2d_transpose_18/mul_1:z:0,model_3/conv2d_transpose_18/stack/3:output:0*
T0*
N*
_output_shapes
:{
1model_3/conv2d_transpose_18/strided_slice_3/stackConst*
valueB: *
_output_shapes
:*
dtype0}
3model_3/conv2d_transpose_18/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:}
3model_3/conv2d_transpose_18/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
+model_3/conv2d_transpose_18/strided_slice_3StridedSlice*model_3/conv2d_transpose_18/stack:output:0:model_3/conv2d_transpose_18/strided_slice_3/stack:output:0<model_3/conv2d_transpose_18/strided_slice_3/stack_1:output:0<model_3/conv2d_transpose_18/strided_slice_3/stack_2:output:0*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask�
;model_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOpVmodel_3_conv2d_transpose_18_conv2d_transpose_readvariableop_conv2d_transpose_18_kernel*&
_output_shapes
:  *
dtype0�
,model_3/conv2d_transpose_18/conv2d_transposeConv2DBackpropInput*model_3/conv2d_transpose_18/stack:output:0Cmodel_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0.model_3/conv2d_transpose_17/Relu:activations:0*
paddingSAME*0
_output_shapes
:���������x� *
strides
*
T0�
2model_3/conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOpKmodel_3_conv2d_transpose_18_biasadd_readvariableop_conv2d_transpose_18_bias*
dtype0*
_output_shapes
: �
#model_3/conv2d_transpose_18/BiasAddBiasAdd5model_3/conv2d_transpose_18/conv2d_transpose:output:0:model_3/conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������x� �
 model_3/conv2d_transpose_18/ReluRelu,model_3/conv2d_transpose_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������x� 
!model_3/conv2d_transpose_19/ShapeShape.model_3/conv2d_transpose_18/Relu:activations:0*
T0*
_output_shapes
:y
/model_3/conv2d_transpose_19/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:{
1model_3/conv2d_transpose_19/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:{
1model_3/conv2d_transpose_19/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:�
)model_3/conv2d_transpose_19/strided_sliceStridedSlice*model_3/conv2d_transpose_19/Shape:output:08model_3/conv2d_transpose_19/strided_slice/stack:output:0:model_3/conv2d_transpose_19/strided_slice/stack_1:output:0:model_3/conv2d_transpose_19/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
shrink_axis_mask*
Index0{
1model_3/conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
valueB:*
dtype0}
3model_3/conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0}
3model_3/conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
+model_3/conv2d_transpose_19/strided_slice_1StridedSlice*model_3/conv2d_transpose_19/Shape:output:0:model_3/conv2d_transpose_19/strided_slice_1/stack:output:0<model_3/conv2d_transpose_19/strided_slice_1/stack_1:output:0<model_3/conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
_output_shapes
: *
T0*
shrink_axis_mask{
1model_3/conv2d_transpose_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model_3/conv2d_transpose_19/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0}
3model_3/conv2d_transpose_19/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
+model_3/conv2d_transpose_19/strided_slice_2StridedSlice*model_3/conv2d_transpose_19/Shape:output:0:model_3/conv2d_transpose_19/strided_slice_2/stack:output:0<model_3/conv2d_transpose_19/strided_slice_2/stack_1:output:0<model_3/conv2d_transpose_19/strided_slice_2/stack_2:output:0*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: c
!model_3/conv2d_transpose_19/mul/yConst*
dtype0*
_output_shapes
: *
value	B :�
model_3/conv2d_transpose_19/mulMul4model_3/conv2d_transpose_19/strided_slice_1:output:0*model_3/conv2d_transpose_19/mul/y:output:0*
_output_shapes
: *
T0e
#model_3/conv2d_transpose_19/mul_1/yConst*
dtype0*
_output_shapes
: *
value	B :�
!model_3/conv2d_transpose_19/mul_1Mul4model_3/conv2d_transpose_19/strided_slice_2:output:0,model_3/conv2d_transpose_19/mul_1/y:output:0*
_output_shapes
: *
T0e
#model_3/conv2d_transpose_19/stack/3Const*
dtype0*
_output_shapes
: *
value	B :�
!model_3/conv2d_transpose_19/stackPack2model_3/conv2d_transpose_19/strided_slice:output:0#model_3/conv2d_transpose_19/mul:z:0%model_3/conv2d_transpose_19/mul_1:z:0,model_3/conv2d_transpose_19/stack/3:output:0*
T0*
_output_shapes
:*
N{
1model_3/conv2d_transpose_19/strided_slice_3/stackConst*
_output_shapes
:*
valueB: *
dtype0}
3model_3/conv2d_transpose_19/strided_slice_3/stack_1Const*
valueB:*
_output_shapes
:*
dtype0}
3model_3/conv2d_transpose_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+model_3/conv2d_transpose_19/strided_slice_3StridedSlice*model_3/conv2d_transpose_19/stack:output:0:model_3/conv2d_transpose_19/strided_slice_3/stack:output:0<model_3/conv2d_transpose_19/strided_slice_3/stack_1:output:0<model_3/conv2d_transpose_19/strided_slice_3/stack_2:output:0*
T0*
_output_shapes
: *
shrink_axis_mask*
Index0�
;model_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOpVmodel_3_conv2d_transpose_19_conv2d_transpose_readvariableop_conv2d_transpose_19_kernel*&
_output_shapes
: *
dtype0�
,model_3/conv2d_transpose_19/conv2d_transposeConv2DBackpropInput*model_3/conv2d_transpose_19/stack:output:0Cmodel_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0.model_3/conv2d_transpose_18/Relu:activations:0*
paddingSAME*1
_output_shapes
:�����������*
strides
*
T0�
2model_3/conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOpKmodel_3_conv2d_transpose_19_biasadd_readvariableop_conv2d_transpose_19_bias*
_output_shapes
:*
dtype0�
#model_3/conv2d_transpose_19/BiasAddBiasAdd5model_3/conv2d_transpose_19/conv2d_transpose:output:0:model_3/conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
 model_3/conv2d_transpose_19/ReluRelu,model_3/conv2d_transpose_19/BiasAdd:output:0*1
_output_shapes
:�����������*
T0�	
IdentityIdentity.model_3/conv2d_transpose_19/Relu:activations:0<^model_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOp&^model_3/dense_4/MatMul/ReadVariableOp&^model_3/z_mean/BiasAdd/ReadVariableOp)^model_3/z_log_var/BiasAdd/ReadVariableOp(^model_3/conv2d_19/Conv2D/ReadVariableOp3^model_3/conv2d_transpose_17/BiasAdd/ReadVariableOp(^model_3/conv2d_17/Conv2D/ReadVariableOp(^model_3/z_log_var/MatMul/ReadVariableOp(^model_3/conv2d_18/Conv2D/ReadVariableOp)^model_3/conv2d_16/BiasAdd/ReadVariableOp(^model_3/conv2d_16/Conv2D/ReadVariableOp)^model_3/conv2d_18/BiasAdd/ReadVariableOp<^model_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOp)^model_3/conv2d_17/BiasAdd/ReadVariableOp3^model_3/conv2d_transpose_16/BiasAdd/ReadVariableOp<^model_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOp<^model_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOp3^model_3/conv2d_transpose_19/BiasAdd/ReadVariableOp%^model_3/z_mean/MatMul/ReadVariableOp'^model_3/dense_4/BiasAdd/ReadVariableOp3^model_3/conv2d_transpose_18/BiasAdd/ReadVariableOp)^model_3/conv2d_19/BiasAdd/ReadVariableOp*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*�
_input_shapesw
u:�����������::::::::::::::::::::::2N
%model_3/z_mean/BiasAdd/ReadVariableOp%model_3/z_mean/BiasAdd/ReadVariableOp2z
;model_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOp;model_3/conv2d_transpose_16/conv2d_transpose/ReadVariableOp2z
;model_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOp;model_3/conv2d_transpose_17/conv2d_transpose/ReadVariableOp2z
;model_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOp;model_3/conv2d_transpose_18/conv2d_transpose/ReadVariableOp2T
(model_3/conv2d_18/BiasAdd/ReadVariableOp(model_3/conv2d_18/BiasAdd/ReadVariableOp2z
;model_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOp;model_3/conv2d_transpose_19/conv2d_transpose/ReadVariableOp2h
2model_3/conv2d_transpose_19/BiasAdd/ReadVariableOp2model_3/conv2d_transpose_19/BiasAdd/ReadVariableOp2R
'model_3/conv2d_19/Conv2D/ReadVariableOp'model_3/conv2d_19/Conv2D/ReadVariableOp2N
%model_3/dense_4/MatMul/ReadVariableOp%model_3/dense_4/MatMul/ReadVariableOp2T
(model_3/conv2d_16/BiasAdd/ReadVariableOp(model_3/conv2d_16/BiasAdd/ReadVariableOp2h
2model_3/conv2d_transpose_17/BiasAdd/ReadVariableOp2model_3/conv2d_transpose_17/BiasAdd/ReadVariableOp2R
'model_3/conv2d_16/Conv2D/ReadVariableOp'model_3/conv2d_16/Conv2D/ReadVariableOp2P
&model_3/dense_4/BiasAdd/ReadVariableOp&model_3/dense_4/BiasAdd/ReadVariableOp2L
$model_3/z_mean/MatMul/ReadVariableOp$model_3/z_mean/MatMul/ReadVariableOp2T
(model_3/conv2d_19/BiasAdd/ReadVariableOp(model_3/conv2d_19/BiasAdd/ReadVariableOp2T
(model_3/conv2d_17/BiasAdd/ReadVariableOp(model_3/conv2d_17/BiasAdd/ReadVariableOp2R
'model_3/conv2d_17/Conv2D/ReadVariableOp'model_3/conv2d_17/Conv2D/ReadVariableOp2T
(model_3/z_log_var/BiasAdd/ReadVariableOp(model_3/z_log_var/BiasAdd/ReadVariableOp2h
2model_3/conv2d_transpose_18/BiasAdd/ReadVariableOp2model_3/conv2d_transpose_18/BiasAdd/ReadVariableOp2R
'model_3/z_log_var/MatMul/ReadVariableOp'model_3/z_log_var/MatMul/ReadVariableOp2h
2model_3/conv2d_transpose_16/BiasAdd/ReadVariableOp2model_3/conv2d_transpose_16/BiasAdd/ReadVariableOp2R
'model_3/conv2d_18/Conv2D/ReadVariableOp'model_3/conv2d_18/Conv2D/ReadVariableOp:' #
!
_user_specified_name	input_5: : : : : : : : :	 :
 : : : : : : : : : : : : 
��
�(
 __inference__traced_restore_8122
file_prefix%
!assignvariableop_conv2d_16_kernel%
!assignvariableop_1_conv2d_16_bias'
#assignvariableop_2_conv2d_17_kernel%
!assignvariableop_3_conv2d_17_bias'
#assignvariableop_4_conv2d_18_kernel%
!assignvariableop_5_conv2d_18_bias'
#assignvariableop_6_conv2d_19_kernel%
!assignvariableop_7_conv2d_19_bias&
"assignvariableop_8_z_mean_4_kernel$
 assignvariableop_9_z_mean_4_bias*
&assignvariableop_10_z_log_var_4_kernel(
$assignvariableop_11_z_log_var_4_bias&
"assignvariableop_12_dense_4_kernel$
 assignvariableop_13_dense_4_bias2
.assignvariableop_14_conv2d_transpose_16_kernel0
,assignvariableop_15_conv2d_transpose_16_bias2
.assignvariableop_16_conv2d_transpose_17_kernel0
,assignvariableop_17_conv2d_transpose_17_bias2
.assignvariableop_18_conv2d_transpose_18_kernel0
,assignvariableop_19_conv2d_transpose_18_bias2
.assignvariableop_20_conv2d_transpose_19_kernel0
,assignvariableop_21_conv2d_transpose_19_bias#
assignvariableop_22_adam_4_iter%
!assignvariableop_23_adam_4_beta_1%
!assignvariableop_24_adam_4_beta_2$
 assignvariableop_25_adam_4_decay,
(assignvariableop_26_adam_4_learning_rate1
-assignvariableop_27_adam_4_conv2d_16_kernel_m/
+assignvariableop_28_adam_4_conv2d_16_bias_m1
-assignvariableop_29_adam_4_conv2d_17_kernel_m/
+assignvariableop_30_adam_4_conv2d_17_bias_m1
-assignvariableop_31_adam_4_conv2d_18_kernel_m/
+assignvariableop_32_adam_4_conv2d_18_bias_m1
-assignvariableop_33_adam_4_conv2d_19_kernel_m/
+assignvariableop_34_adam_4_conv2d_19_bias_m0
,assignvariableop_35_adam_4_z_mean_4_kernel_m.
*assignvariableop_36_adam_4_z_mean_4_bias_m3
/assignvariableop_37_adam_4_z_log_var_4_kernel_m1
-assignvariableop_38_adam_4_z_log_var_4_bias_m/
+assignvariableop_39_adam_4_dense_4_kernel_m-
)assignvariableop_40_adam_4_dense_4_bias_m;
7assignvariableop_41_adam_4_conv2d_transpose_16_kernel_m9
5assignvariableop_42_adam_4_conv2d_transpose_16_bias_m;
7assignvariableop_43_adam_4_conv2d_transpose_17_kernel_m9
5assignvariableop_44_adam_4_conv2d_transpose_17_bias_m;
7assignvariableop_45_adam_4_conv2d_transpose_18_kernel_m9
5assignvariableop_46_adam_4_conv2d_transpose_18_bias_m;
7assignvariableop_47_adam_4_conv2d_transpose_19_kernel_m9
5assignvariableop_48_adam_4_conv2d_transpose_19_bias_m1
-assignvariableop_49_adam_4_conv2d_16_kernel_v/
+assignvariableop_50_adam_4_conv2d_16_bias_v1
-assignvariableop_51_adam_4_conv2d_17_kernel_v/
+assignvariableop_52_adam_4_conv2d_17_bias_v1
-assignvariableop_53_adam_4_conv2d_18_kernel_v/
+assignvariableop_54_adam_4_conv2d_18_bias_v1
-assignvariableop_55_adam_4_conv2d_19_kernel_v/
+assignvariableop_56_adam_4_conv2d_19_bias_v0
,assignvariableop_57_adam_4_z_mean_4_kernel_v.
*assignvariableop_58_adam_4_z_mean_4_bias_v3
/assignvariableop_59_adam_4_z_log_var_4_kernel_v1
-assignvariableop_60_adam_4_z_log_var_4_bias_v/
+assignvariableop_61_adam_4_dense_4_kernel_v-
)assignvariableop_62_adam_4_dense_4_bias_v;
7assignvariableop_63_adam_4_conv2d_transpose_16_kernel_v9
5assignvariableop_64_adam_4_conv2d_transpose_16_bias_v;
7assignvariableop_65_adam_4_conv2d_transpose_17_kernel_v9
5assignvariableop_66_adam_4_conv2d_transpose_17_bias_v;
7assignvariableop_67_adam_4_conv2d_transpose_18_kernel_v9
5assignvariableop_68_adam_4_conv2d_transpose_18_bias_v;
7assignvariableop_69_adam_4_conv2d_transpose_19_kernel_v9
5assignvariableop_70_adam_4_conv2d_transpose_19_bias_v
identity_72��AssignVariableOp_35�AssignVariableOp_54�AssignVariableOp_1�AssignVariableOp_20�AssignVariableOp_39�AssignVariableOp_58�AssignVariableOp_5�AssignVariableOp_24�AssignVariableOp_43�AssignVariableOp_62�AssignVariableOp_9�AssignVariableOp_28�AssignVariableOp_47�AssignVariableOp_66�AssignVariableOp_13�AssignVariableOp_32�AssignVariableOp_51�AssignVariableOp_70�AssignVariableOp_17�AssignVariableOp_36�AssignVariableOp_65�AssignVariableOp_55�AssignVariableOp_2�AssignVariableOp_21�AssignVariableOp_40�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_25�RestoreV2_1�AssignVariableOp_44�AssignVariableOp_63�AssignVariableOp_10�AssignVariableOp_29�	RestoreV2�AssignVariableOp_48�AssignVariableOp_12�AssignVariableOp_67�AssignVariableOp_14�AssignVariableOp_33�AssignVariableOp_52�AssignVariableOp_18�AssignVariableOp_37�AssignVariableOp_56�AssignVariableOp_3�AssignVariableOp_22�AssignVariableOp_41�AssignVariableOp_60�AssignVariableOp_7�AssignVariableOp_26�AssignVariableOp_45�AssignVariableOp_31�AssignVariableOp_64�AssignVariableOp_11�AssignVariableOp_30�AssignVariableOp_49�AssignVariableOp_50�AssignVariableOp_68�AssignVariableOp_15�AssignVariableOp_34�AssignVariableOp_53�AssignVariableOp�AssignVariableOp_19�AssignVariableOp_38�AssignVariableOp_57�AssignVariableOp_4�AssignVariableOp_23�AssignVariableOp_42�AssignVariableOp_16�AssignVariableOp_61�AssignVariableOp_8�AssignVariableOp_27�AssignVariableOp_46�AssignVariableOp_69�(
RestoreV2/tensor_namesConst"/device:CPU:0*�'
value�'B�'GB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:G�
RestoreV2/shape_and_slicesConst"/device:CPU:0*�
value�B�GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:G�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*U
dtypesK
I2G	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:}
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_16_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_16_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_17_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_17_biasIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_18_kernelIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_18_biasIdentity_5:output:0*
_output_shapes
 *
dtype0N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_19_kernelIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_19_biasIdentity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_z_mean_4_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_z_mean_4_biasIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_z_log_var_4_kernelIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_z_log_var_4_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_4_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_4_biasIdentity_13:output:0*
_output_shapes
 *
dtype0P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp.assignvariableop_14_conv2d_transpose_16_kernelIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_conv2d_transpose_16_biasIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp.assignvariableop_16_conv2d_transpose_17_kernelIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp,assignvariableop_17_conv2d_transpose_17_biasIdentity_17:output:0*
_output_shapes
 *
dtype0P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0�
AssignVariableOp_18AssignVariableOp.assignvariableop_18_conv2d_transpose_18_kernelIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp,assignvariableop_19_conv2d_transpose_18_biasIdentity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp.assignvariableop_20_conv2d_transpose_19_kernelIdentity_20:output:0*
_output_shapes
 *
dtype0P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp,assignvariableop_21_conv2d_transpose_19_biasIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0	*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_4_iterIdentity_22:output:0*
_output_shapes
 *
dtype0	P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_adam_4_beta_1Identity_23:output:0*
_output_shapes
 *
dtype0P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0�
AssignVariableOp_24AssignVariableOp!assignvariableop_24_adam_4_beta_2Identity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0�
AssignVariableOp_25AssignVariableOp assignvariableop_25_adam_4_decayIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_4_learning_rateIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0�
AssignVariableOp_27AssignVariableOp-assignvariableop_27_adam_4_conv2d_16_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype0P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_4_conv2d_16_bias_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp-assignvariableop_29_adam_4_conv2d_17_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype0P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_4_conv2d_17_bias_mIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
_output_shapes
:*
T0�
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_4_conv2d_18_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype0P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_4_conv2d_18_bias_mIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp-assignvariableop_33_adam_4_conv2d_19_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype0P
Identity_34IdentityRestoreV2:tensors:34*
_output_shapes
:*
T0�
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_4_conv2d_19_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype0P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_4_z_mean_4_kernel_mIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
_output_shapes
:*
T0�
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_4_z_mean_4_bias_mIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
_output_shapes
:*
T0�
AssignVariableOp_37AssignVariableOp/assignvariableop_37_adam_4_z_log_var_4_kernel_mIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp-assignvariableop_38_adam_4_z_log_var_4_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype0P
Identity_39IdentityRestoreV2:tensors:39*
_output_shapes
:*
T0�
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_4_dense_4_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype0P
Identity_40IdentityRestoreV2:tensors:40*
_output_shapes
:*
T0�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_4_dense_4_bias_mIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_4_conv2d_transpose_16_kernel_mIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_4_conv2d_transpose_16_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype0P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_4_conv2d_transpose_17_kernel_mIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
_output_shapes
:*
T0�
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_4_conv2d_transpose_17_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype0P
Identity_45IdentityRestoreV2:tensors:45*
_output_shapes
:*
T0�
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_4_conv2d_transpose_18_kernel_mIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_4_conv2d_transpose_18_bias_mIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
_output_shapes
:*
T0�
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_4_conv2d_transpose_19_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype0P
Identity_48IdentityRestoreV2:tensors:48*
_output_shapes
:*
T0�
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_4_conv2d_transpose_19_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype0P
Identity_49IdentityRestoreV2:tensors:49*
_output_shapes
:*
T0�
AssignVariableOp_49AssignVariableOp-assignvariableop_49_adam_4_conv2d_16_kernel_vIdentity_49:output:0*
_output_shapes
 *
dtype0P
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp+assignvariableop_50_adam_4_conv2d_16_bias_vIdentity_50:output:0*
_output_shapes
 *
dtype0P
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adam_4_conv2d_17_kernel_vIdentity_51:output:0*
_output_shapes
 *
dtype0P
Identity_52IdentityRestoreV2:tensors:52*
_output_shapes
:*
T0�
AssignVariableOp_52AssignVariableOp+assignvariableop_52_adam_4_conv2d_17_bias_vIdentity_52:output:0*
dtype0*
_output_shapes
 P
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp-assignvariableop_53_adam_4_conv2d_18_kernel_vIdentity_53:output:0*
dtype0*
_output_shapes
 P
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp+assignvariableop_54_adam_4_conv2d_18_bias_vIdentity_54:output:0*
dtype0*
_output_shapes
 P
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp-assignvariableop_55_adam_4_conv2d_19_kernel_vIdentity_55:output:0*
dtype0*
_output_shapes
 P
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp+assignvariableop_56_adam_4_conv2d_19_bias_vIdentity_56:output:0*
dtype0*
_output_shapes
 P
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_4_z_mean_4_kernel_vIdentity_57:output:0*
dtype0*
_output_shapes
 P
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_4_z_mean_4_bias_vIdentity_58:output:0*
_output_shapes
 *
dtype0P
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp/assignvariableop_59_adam_4_z_log_var_4_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype0P
Identity_60IdentityRestoreV2:tensors:60*
_output_shapes
:*
T0�
AssignVariableOp_60AssignVariableOp-assignvariableop_60_adam_4_z_log_var_4_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype0P
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_4_dense_4_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype0P
Identity_62IdentityRestoreV2:tensors:62*
_output_shapes
:*
T0�
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_4_dense_4_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype0P
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_4_conv2d_transpose_16_kernel_vIdentity_63:output:0*
dtype0*
_output_shapes
 P
Identity_64IdentityRestoreV2:tensors:64*
_output_shapes
:*
T0�
AssignVariableOp_64AssignVariableOp5assignvariableop_64_adam_4_conv2d_transpose_16_bias_vIdentity_64:output:0*
dtype0*
_output_shapes
 P
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp7assignvariableop_65_adam_4_conv2d_transpose_17_kernel_vIdentity_65:output:0*
dtype0*
_output_shapes
 P
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_4_conv2d_transpose_17_bias_vIdentity_66:output:0*
_output_shapes
 *
dtype0P
Identity_67IdentityRestoreV2:tensors:67*
_output_shapes
:*
T0�
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_4_conv2d_transpose_18_kernel_vIdentity_67:output:0*
dtype0*
_output_shapes
 P
Identity_68IdentityRestoreV2:tensors:68*
_output_shapes
:*
T0�
AssignVariableOp_68AssignVariableOp5assignvariableop_68_adam_4_conv2d_transpose_18_bias_vIdentity_68:output:0*
dtype0*
_output_shapes
 P
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_4_conv2d_transpose_19_kernel_vIdentity_69:output:0*
_output_shapes
 *
dtype0P
Identity_70IdentityRestoreV2:tensors:70*
_output_shapes
:*
T0�
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_4_conv2d_transpose_19_bias_vIdentity_70:output:0*
_output_shapes
 *
dtype0�
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_71Identityfile_prefix^AssignVariableOp_35^AssignVariableOp_43^AssignVariableOp_50^AssignVariableOp_54^AssignVariableOp_62^AssignVariableOp_6^AssignVariableOp_12^AssignVariableOp_51^AssignVariableOp_55^AssignVariableOp_63^AssignVariableOp_70^AssignVariableOp_56^AssignVariableOp_64^NoOp^AssignVariableOp_57^AssignVariableOp_65^AssignVariableOp_58^AssignVariableOp_66^AssignVariableOp_59^AssignVariableOp_18^AssignVariableOp_22^AssignVariableOp_34^AssignVariableOp_42^AssignVariableOp_49^AssignVariableOp_28^AssignVariableOp_13^AssignVariableOp_23^AssignVariableOp_29^AssignVariableOp_36^AssignVariableOp_14^AssignVariableOp_20^AssignVariableOp_30^AssignVariableOp_37^AssignVariableOp_44^AssignVariableOp_16^AssignVariableOp_21^AssignVariableOp_31^AssignVariableOp_38^AssignVariableOp_45^AssignVariableOp_52^AssignVariableOp_19^AssignVariableOp_24^AssignVariableOp_32^AssignVariableOp_39^AssignVariableOp_46^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_15^AssignVariableOp_25^AssignVariableOp_7^AssignVariableOp_40^AssignVariableOp_4^AssignVariableOp_8^AssignVariableOp_60^AssignVariableOp_67^AssignVariableOp_26^AssignVariableOp_33^AssignVariableOp_41^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_9^AssignVariableOp_68^AssignVariableOp_17^AssignVariableOp_27^AssignVariableOp^AssignVariableOp_10^AssignVariableOp_48^AssignVariableOp_53^AssignVariableOp_61^AssignVariableOp_69^AssignVariableOp_3^AssignVariableOp_11"/device:CPU:0*
T0*
_output_shapes
: �
Identity_72IdentityIdentity_71:output:0^AssignVariableOp_59^AssignVariableOp_48^AssignVariableOp_37^AssignVariableOp_41^AssignVariableOp_49^AssignVariableOp_19^AssignVariableOp_16^AssignVariableOp_43^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_50^AssignVariableOp_38^AssignVariableOp_61^AssignVariableOp_58^AssignVariableOp_31^AssignVariableOp_68^AssignVariableOp_57^AssignVariableOp_8^AssignVariableOp_15^AssignVariableOp_4^AssignVariableOp_27^AssignVariableOp_34^AssignVariableOp_23^AssignVariableOp_46^AssignVariableOp_69^AssignVariableOp_9^AssignVariableOp_66^AssignVariableOp_55^AssignVariableOp_29^AssignVariableOp_52^AssignVariableOp_60
^RestoreV2^AssignVariableOp_18^AssignVariableOp_26^AssignVariableOp_35^AssignVariableOp_56^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_25^AssignVariableOp_54^AssignVariableOp_5^AssignVariableOp_70^AssignVariableOp_21^RestoreV2_1^AssignVariableOp_12^AssignVariableOp_1^AssignVariableOp_24^AssignVariableOp_17^AssignVariableOp_36^AssignVariableOp_44^AssignVariableOp_67^AssignVariableOp_3^AssignVariableOp_64^AssignVariableOp_20^AssignVariableOp_32^AssignVariableOp_65^AssignVariableOp_63^AssignVariableOp_14^AssignVariableOp_45^AssignVariableOp_11^AssignVariableOp_53^AssignVariableOp_39^AssignVariableOp_28^AssignVariableOp_40^AssignVariableOp_10^AssignVariableOp_33^AssignVariableOp_22^AssignVariableOp_30^AssignVariableOp^AssignVariableOp_42^AssignVariableOp_62^AssignVariableOp_47*
T0*
_output_shapes
: "#
identity_72Identity_72:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_59AssignVariableOp_592*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_23: : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G :+ '
%
_user_specified_namefile_prefix: : : : "&L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_5:
serving_default_input_5:0�����������Q
conv2d_transpose_19:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:�+
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
	optimizer

signatures
Y_default_save_signature"
_generic_user_object
"
_generic_user_object
4

kernel
bias"
_generic_user_object
4

kernel
bias"
_generic_user_object
4

kernel
bias"
_generic_user_object
4

kernel
bias"
_generic_user_object
"
_generic_user_object
4

kernel
bias"
_generic_user_object
4

kernel
bias"
_generic_user_object
"
_generic_user_object
4

kernel
bias"
_generic_user_object
"
_generic_user_object
4

 kernel
!bias"
_generic_user_object
4

"kernel
#bias"
_generic_user_object
4

$kernel
%bias"
_generic_user_object
4

&kernel
'bias"
_generic_user_object
�
(iter

)beta_1

*beta_2
	+decay
,learning_ratem-m.m/m0m1m2m3m4m5m6m7m8m9m: m;!m<"m=#m>$m?%m@&mA'mBvCvDvEvFvGvHvIvJvKvLvMvNvOvP vQ!vR"vS#vT$vU%vV&vW'vX"
	optimizer
,
Zserving_default"
signature_map
*:( 2conv2d_16/kernel
: 2conv2d_16/bias
*:(  2conv2d_17/kernel
: 2conv2d_17/bias
*:( @2conv2d_18/kernel
:@2conv2d_18/bias
*:(@@2conv2d_19/kernel
:@2conv2d_19/bias
#:!
��2z_mean_4/kernel
:�2z_mean_4/bias
&:$
��2z_log_var_4/kernel
:�2z_log_var_4/bias
": 
��2dense_4/kernel
:�2dense_4/bias
4:2@@2conv2d_transpose_16/kernel
&:$@2conv2d_transpose_16/bias
4:2 @2conv2d_transpose_17/kernel
&:$ 2conv2d_transpose_17/bias
4:2  2conv2d_transpose_18/kernel
&:$ 2conv2d_transpose_18/bias
4:2 2conv2d_transpose_19/kernel
&:$2conv2d_transpose_19/bias
:	 (2Adam_4/iter
: (2Adam_4/beta_1
: (2Adam_4/beta_2
: (2Adam_4/decay
: (2Adam_4/learning_rate
1:/ 2Adam_4/conv2d_16/kernel/m
#:! 2Adam_4/conv2d_16/bias/m
1:/  2Adam_4/conv2d_17/kernel/m
#:! 2Adam_4/conv2d_17/bias/m
1:/ @2Adam_4/conv2d_18/kernel/m
#:!@2Adam_4/conv2d_18/bias/m
1:/@@2Adam_4/conv2d_19/kernel/m
#:!@2Adam_4/conv2d_19/bias/m
*:(
��2Adam_4/z_mean_4/kernel/m
#:!�2Adam_4/z_mean_4/bias/m
-:+
��2Adam_4/z_log_var_4/kernel/m
&:$�2Adam_4/z_log_var_4/bias/m
):'
��2Adam_4/dense_4/kernel/m
": �2Adam_4/dense_4/bias/m
;:9@@2#Adam_4/conv2d_transpose_16/kernel/m
-:+@2!Adam_4/conv2d_transpose_16/bias/m
;:9 @2#Adam_4/conv2d_transpose_17/kernel/m
-:+ 2!Adam_4/conv2d_transpose_17/bias/m
;:9  2#Adam_4/conv2d_transpose_18/kernel/m
-:+ 2!Adam_4/conv2d_transpose_18/bias/m
;:9 2#Adam_4/conv2d_transpose_19/kernel/m
-:+2!Adam_4/conv2d_transpose_19/bias/m
1:/ 2Adam_4/conv2d_16/kernel/v
#:! 2Adam_4/conv2d_16/bias/v
1:/  2Adam_4/conv2d_17/kernel/v
#:! 2Adam_4/conv2d_17/bias/v
1:/ @2Adam_4/conv2d_18/kernel/v
#:!@2Adam_4/conv2d_18/bias/v
1:/@@2Adam_4/conv2d_19/kernel/v
#:!@2Adam_4/conv2d_19/bias/v
*:(
��2Adam_4/z_mean_4/kernel/v
#:!�2Adam_4/z_mean_4/bias/v
-:+
��2Adam_4/z_log_var_4/kernel/v
&:$�2Adam_4/z_log_var_4/bias/v
):'
��2Adam_4/dense_4/kernel/v
": �2Adam_4/dense_4/bias/v
;:9@@2#Adam_4/conv2d_transpose_16/kernel/v
-:+@2!Adam_4/conv2d_transpose_16/bias/v
;:9 @2#Adam_4/conv2d_transpose_17/kernel/v
-:+ 2!Adam_4/conv2d_transpose_17/bias/v
;:9  2#Adam_4/conv2d_transpose_18/kernel/v
-:+ 2!Adam_4/conv2d_transpose_18/bias/v
;:9 2#Adam_4/conv2d_transpose_19/kernel/v
-:+2!Adam_4/conv2d_transpose_19/bias/v
�2�
__inference__wrapped_model_7626�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *0�-
+�(
input_5�����������
1B/
"__inference_signature_wrapper_7657input_5�
"__inference_signature_wrapper_7657� !"#$%&'E�B
� 
;�8
6
input_5+�(
input_5�����������"S�P
N
conv2d_transpose_197�4
conv2d_transpose_19������������
__inference__wrapped_model_7626� !"#$%&':�7
0�-
+�(
input_5�����������
� "S�P
N
conv2d_transpose_197�4
conv2d_transpose_19�����������