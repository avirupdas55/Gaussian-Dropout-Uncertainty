ќБ

иЎ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02v2.7.0-0-gc256c071bb28Фт
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:2*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:2*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ШLє*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
ШLє*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:є*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:є*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	є
*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	є
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:2*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:2*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ШLє*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
ШLє*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:є*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:є*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	є
*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	є
*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:
*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:2*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:2*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ШLє*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
ШLє*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:є*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:є*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	є
*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	є
*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
ч;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ђ;
value;B; B;
л
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
R
)	variables
*trainable_variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
R
3	variables
4trainable_variables
5regularization_losses
6	keras_api
h

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
R
=	variables
>trainable_variables
?regularization_losses
@	keras_api
р
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratemmm m-m.m7m8mvvv v-v.v7v8v
8
0
1
2
 3
-4
.5
76
87
8
0
1
2
 3
-4
.5
76
87
 
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
­
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
­
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
­
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
!	variables
"trainable_variables
#regularization_losses
 
 
 
­
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
%	variables
&trainable_variables
'regularization_losses
 
 
 
­
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
)	variables
*trainable_variables
+regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
/	variables
0trainable_variables
1regularization_losses
 
 
 
­
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
3	variables
4trainable_variables
5regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81

70
81
 
­
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
9	variables
:trainable_variables
;regularization_losses
 
 
 
­
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
=	variables
>trainable_variables
?regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
F
0
1
2
3
4
5
6
7
	8

9

}0
~1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
7
	total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv2d_inputPlaceholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ
С
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *,
f'R%
#__inference_signature_wrapper_75950
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *'
f"R 
__inference__traced_save_76448
ю
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 **
f%R#
!__inference__traced_restore_76557ѕа

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75484

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ю
a
E__inference_activation_layer_call_and_return_conditional_losses_75620

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:џџџџџџџџџ
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ
:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
ы

&__inference_conv2d_layer_call_fn_76115

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_75504w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы	
Э
*__inference_sequential_layer_call_fn_75992

inputs!
unknown:
	unknown_0:#
	unknown_1:2
	unknown_2:2
	unknown_3:
ШLє
	unknown_4:	є
	unknown_5:	є

	unknown_6:

identityЂStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_75821o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ќ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_76198

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0{
BiasAddAddV2Conv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2b
IdentityIdentityBiasAdd:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
БF

__inference__traced_save_76448
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Џ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*и
valueЮBЫ"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHБ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B §
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ї
_input_shapes
: :::2:2:
ШLє:є:	є
:
: : : : : : : : : :::2:2:
ШLє:є:	є
:
:::2:2:
ШLє:є:	є
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:2: 

_output_shapes
:2:&"
 
_output_shapes
:
ШLє:!

_output_shapes	
:є:%!

_output_shapes
:	є
: 

_output_shapes
:
:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:2: 

_output_shapes
:2:&"
 
_output_shapes
:
ШLє:!

_output_shapes	
:є:%!

_output_shapes
:	є
: 

_output_shapes
:
:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:2: 

_output_shapes
:2:&"
 
_output_shapes
:
ШLє:!

_output_shapes	
:є:% !

_output_shapes
:	є
: !

_output_shapes
:
:"

_output_shapes
: 
Й
I
-__inference_max_pooling2d_layer_call_fn_76164

inputs
identityн
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75484
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І
F
*__inference_activation_layer_call_fn_76321

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_75620`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ
:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
НA
И
E__inference_sequential_layer_call_and_return_conditional_losses_76049

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:26
(conv2d_1_biasadd_readvariableop_resource:28
$dense_matmul_readvariableop_resource:
ШLє4
%dense_biasadd_readvariableop_resource:	є9
&dense_1_matmul_readvariableop_resource:	є
5
'dense_1_biasadd_readvariableop_resource:

identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ї
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddAddV2conv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ]
my_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_dropout/dropout/MulMulconv2d/BiasAdd:z:0!my_dropout/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџZ
my_dropout/dropout/ShapeShapeconv2d/BiasAdd:z:0*
T0*
_output_shapes
:Ц
/my_dropout/dropout/random_uniform/RandomUniformRandomUniform!my_dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed**
seed2ыхЯf
!my_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Я
my_dropout/dropout/GreaterEqualGreaterEqual8my_dropout/dropout/random_uniform/RandomUniform:output:0*my_dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
my_dropout/dropout/CastCast#my_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ
my_dropout/dropout/Mul_1Mulmy_dropout/dropout/Mul:z:0my_dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџЋ
max_pooling2d/MaxPoolMaxPoolmy_dropout/dropout/Mul_1:z:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0У
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
conv2d_1/BiasAddAddV2conv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2_
my_dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_dropout_1/dropout/MulMulconv2d_1/BiasAdd:z:0#my_dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2^
my_dropout_1/dropout/ShapeShapeconv2d_1/BiasAdd:z:0*
T0*
_output_shapes
:Ъ
1my_dropout_1/dropout/random_uniform/RandomUniformRandomUniform#my_dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
dtype0*

seed**
seed2АќЗh
#my_dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?е
!my_dropout_1/dropout/GreaterEqualGreaterEqual:my_dropout_1/dropout/random_uniform/RandomUniform:output:0,my_dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
my_dropout_1/dropout/CastCast%my_dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
my_dropout_1/dropout/Mul_1Mulmy_dropout_1/dropout/Mul:z:0my_dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH&  
flatten/ReshapeReshapemy_dropout_1/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџШL
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ШLє*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0
dense/BiasAddAddV2dense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєX

dense/ReluReludense/BiasAdd:z:0*
T0*(
_output_shapes
:џџџџџџџџџє_
my_dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_dropout_2/dropout/MulMuldense/Relu:activations:0#my_dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџєb
my_dropout_2/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:У
1my_dropout_2/dropout/random_uniform/RandomUniformRandomUniform#my_dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџє*
dtype0*

seed**
seed2кУоh
#my_dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ю
!my_dropout_2/dropout/GreaterEqualGreaterEqual:my_dropout_2/dropout/random_uniform/RandomUniform:output:0,my_dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџє
my_dropout_2/dropout/CastCast%my_dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџє
my_dropout_2/dropout/Mul_1Mulmy_dropout_2/dropout/Mul:z:0my_dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	є
*
dtype0
dense_1/MatMulMatMulmy_dropout_2/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_1/BiasAddAddV2dense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
d
activation/SoftmaxSoftmaxdense_1/BiasAdd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
k
IdentityIdentityactivation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76179

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Т	
є
B__inference_dense_1_layer_call_and_return_conditional_losses_76316

inputs1
matmul_readvariableop_resource:	є
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	є
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0t
BiasAddAddV2MatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Z
IdentityIdentityBiasAdd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџє: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs


ќ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75540

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0{
BiasAddAddV2Conv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2b
IdentityIdentityBiasAdd:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а

f
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_76232

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Џ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
dtype0*

seed**
seed2ВЄw[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2:W S
/
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ж,
х
E__inference_sequential_layer_call_and_return_conditional_losses_75821

inputs&
conv2d_75794:
conv2d_75796:(
conv2d_1_75801:2
conv2d_1_75803:2
dense_75808:
ШLє
dense_75810:	є 
dense_1_75814:	є

dense_1_75816:

identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂ"my_dropout/StatefulPartitionedCallЂ$my_dropout_1/StatefulPartitionedCallЂ$my_dropout_2/StatefulPartitionedCallє
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_75794conv2d_75796*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_75504ћ
"my_dropout/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_my_dropout_layer_call_and_return_conditional_losses_75755ѕ
max_pooling2d/PartitionedCallPartitionedCall+my_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75528
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_75801conv2d_1_75803*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75540І
$my_dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0#^my_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_75717ф
flatten/PartitionedCallPartitionedCall-my_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџШL* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75566
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_75808dense_75810*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75579
$my_dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0%^my_dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_75678
dense_1/StatefulPartitionedCallStatefulPartitionedCall-my_dropout_2/StatefulPartitionedCall:output:0dense_1_75814dense_1_75816*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75609ф
activation/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_75620r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
П
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall#^my_dropout/StatefulPartitionedCall%^my_dropout_1/StatefulPartitionedCall%^my_dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"my_dropout/StatefulPartitionedCall"my_dropout/StatefulPartitionedCall2L
$my_dropout_1/StatefulPartitionedCall$my_dropout_1/StatefulPartitionedCall2L
$my_dropout_2/StatefulPartitionedCall$my_dropout_2/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ќ
с
!__inference__traced_restore_76557
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel:2.
 assignvariableop_3_conv2d_1_bias:23
assignvariableop_4_dense_kernel:
ШLє,
assignvariableop_5_dense_bias:	є4
!assignvariableop_6_dense_1_kernel:	є
-
assignvariableop_7_dense_1_bias:
&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: B
(assignvariableop_17_adam_conv2d_kernel_m:4
&assignvariableop_18_adam_conv2d_bias_m:D
*assignvariableop_19_adam_conv2d_1_kernel_m:26
(assignvariableop_20_adam_conv2d_1_bias_m:2;
'assignvariableop_21_adam_dense_kernel_m:
ШLє4
%assignvariableop_22_adam_dense_bias_m:	є<
)assignvariableop_23_adam_dense_1_kernel_m:	є
5
'assignvariableop_24_adam_dense_1_bias_m:
B
(assignvariableop_25_adam_conv2d_kernel_v:4
&assignvariableop_26_adam_conv2d_bias_v:D
*assignvariableop_27_adam_conv2d_1_kernel_v:26
(assignvariableop_28_adam_conv2d_1_bias_v:2;
'assignvariableop_29_adam_dense_kernel_v:
ШLє4
%assignvariableop_30_adam_dense_bias_v:	є<
)assignvariableop_31_adam_dense_1_kernel_v:	є
5
'assignvariableop_32_adam_dense_1_bias_v:

identity_34ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9В
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*и
valueЮBЫ"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHД
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_conv2d_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_conv2d_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv2d_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_conv2d_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dense_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ѕ
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

c
*__inference_my_dropout_layer_call_fn_76135

inputs
identityЂStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_my_dropout_layer_call_and_return_conditional_losses_75755w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


є
@__inference_dense_layer_call_and_return_conditional_losses_76263

inputs2
matmul_readvariableop_resource:
ШLє.
biasadd_readvariableop_resource:	є
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ШLє*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0u
BiasAddAddV2MatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєL
ReluReluBiasAdd:z:0*
T0*(
_output_shapes
:џџџџџџџџџєb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџєw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџШL: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџШL
 
_user_specified_nameinputs


f
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_75678

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџєC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ј
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџє*
dtype0*

seed**
seed2 M[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџєp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџєj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџєZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџє"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџє:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
Ф
^
B__inference_flatten_layer_call_and_return_conditional_losses_76243

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH&  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџШLY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџШL"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2:W S
/
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs

e
,__inference_my_dropout_2_layer_call_fn_76268

inputs
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_75597p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџє`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџє22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
Ю

d
E__inference_my_dropout_layer_call_and_return_conditional_losses_76159

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Џ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed**
seed2Ў[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџa
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
НA
И
E__inference_sequential_layer_call_and_return_conditional_losses_76106

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:26
(conv2d_1_biasadd_readvariableop_resource:28
$dense_matmul_readvariableop_resource:
ШLє4
%dense_biasadd_readvariableop_resource:	є9
&dense_1_matmul_readvariableop_resource:	є
5
'dense_1_biasadd_readvariableop_resource:

identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ї
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddAddV2conv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ]
my_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_dropout/dropout/MulMulconv2d/BiasAdd:z:0!my_dropout/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџZ
my_dropout/dropout/ShapeShapeconv2d/BiasAdd:z:0*
T0*
_output_shapes
:Ц
/my_dropout/dropout/random_uniform/RandomUniformRandomUniform!my_dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed**
seed2ўѕf
!my_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Я
my_dropout/dropout/GreaterEqualGreaterEqual8my_dropout/dropout/random_uniform/RandomUniform:output:0*my_dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
my_dropout/dropout/CastCast#my_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ
my_dropout/dropout/Mul_1Mulmy_dropout/dropout/Mul:z:0my_dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџЋ
max_pooling2d/MaxPoolMaxPoolmy_dropout/dropout/Mul_1:z:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0У
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
conv2d_1/BiasAddAddV2conv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2_
my_dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_dropout_1/dropout/MulMulconv2d_1/BiasAdd:z:0#my_dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2^
my_dropout_1/dropout/ShapeShapeconv2d_1/BiasAdd:z:0*
T0*
_output_shapes
:Ъ
1my_dropout_1/dropout/random_uniform/RandomUniformRandomUniform#my_dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
dtype0*

seed**
seed2Л h
#my_dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?е
!my_dropout_1/dropout/GreaterEqualGreaterEqual:my_dropout_1/dropout/random_uniform/RandomUniform:output:0,my_dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
my_dropout_1/dropout/CastCast%my_dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
my_dropout_1/dropout/Mul_1Mulmy_dropout_1/dropout/Mul:z:0my_dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH&  
flatten/ReshapeReshapemy_dropout_1/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџШL
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ШLє*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0
dense/BiasAddAddV2dense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєX

dense/ReluReludense/BiasAdd:z:0*
T0*(
_output_shapes
:џџџџџџџџџє_
my_dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_dropout_2/dropout/MulMuldense/Relu:activations:0#my_dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџєb
my_dropout_2/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:У
1my_dropout_2/dropout/random_uniform/RandomUniformRandomUniform#my_dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџє*
dtype0*

seed**
seed2Юлh
#my_dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ю
!my_dropout_2/dropout/GreaterEqualGreaterEqual:my_dropout_2/dropout/random_uniform/RandomUniform:output:0,my_dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџє
my_dropout_2/dropout/CastCast%my_dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџє
my_dropout_2/dropout/Mul_1Mulmy_dropout_2/dropout/Mul:z:0my_dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџє
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	є
*
dtype0
dense_1/MatMulMatMulmy_dropout_2/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_1/BiasAddAddV2dense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
d
activation/SoftmaxSoftmaxdense_1/BiasAdd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
k
IdentityIdentityactivation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

c
*__inference_my_dropout_layer_call_fn_76130

inputs
identityЂStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_my_dropout_layer_call_and_return_conditional_losses_75522w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я

d
E__inference_my_dropout_layer_call_and_return_conditional_losses_75755

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed**
seed2рй[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџa
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ш

%__inference_dense_layer_call_fn_76252

inputs
unknown:
ШLє
	unknown_0:	є
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75579p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџє`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџШL: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџШL
 
_user_specified_nameinputs
Ж,
х
E__inference_sequential_layer_call_and_return_conditional_losses_75623

inputs&
conv2d_75505:
conv2d_75507:(
conv2d_1_75541:2
conv2d_1_75543:2
dense_75580:
ШLє
dense_75582:	є 
dense_1_75610:	є

dense_1_75612:

identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂ"my_dropout/StatefulPartitionedCallЂ$my_dropout_1/StatefulPartitionedCallЂ$my_dropout_2/StatefulPartitionedCallє
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_75505conv2d_75507*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_75504ћ
"my_dropout/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_my_dropout_layer_call_and_return_conditional_losses_75522ѕ
max_pooling2d/PartitionedCallPartitionedCall+my_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75528
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_75541conv2d_1_75543*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75540І
$my_dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0#^my_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_75558ф
flatten/PartitionedCallPartitionedCall-my_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџШL* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75566
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_75580dense_75582*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75579
$my_dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0%^my_dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_75597
dense_1/StatefulPartitionedCallStatefulPartitionedCall-my_dropout_2/StatefulPartitionedCall:output:0dense_1_75610dense_1_75612*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75609ф
activation/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_75620r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
П
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall#^my_dropout/StatefulPartitionedCall%^my_dropout_1/StatefulPartitionedCall%^my_dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"my_dropout/StatefulPartitionedCall"my_dropout/StatefulPartitionedCall2L
$my_dropout_1/StatefulPartitionedCall$my_dropout_1/StatefulPartitionedCall2L
$my_dropout_2/StatefulPartitionedCall$my_dropout_2/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


є
@__inference_dense_layer_call_and_return_conditional_losses_75579

inputs2
matmul_readvariableop_resource:
ШLє.
biasadd_readvariableop_resource:	є
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ШLє*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0u
BiasAddAddV2MatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєL
ReluReluBiasAdd:z:0*
T0*(
_output_shapes
:џџџџџџџџџєb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџєw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџШL: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџШL
 
_user_specified_nameinputs
Ш,
ы
E__inference_sequential_layer_call_and_return_conditional_losses_75891
conv2d_input&
conv2d_75864:
conv2d_75866:(
conv2d_1_75871:2
conv2d_1_75873:2
dense_75878:
ШLє
dense_75880:	є 
dense_1_75884:	є

dense_1_75886:

identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂ"my_dropout/StatefulPartitionedCallЂ$my_dropout_1/StatefulPartitionedCallЂ$my_dropout_2/StatefulPartitionedCallњ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_75864conv2d_75866*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_75504ћ
"my_dropout/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_my_dropout_layer_call_and_return_conditional_losses_75522ѕ
max_pooling2d/PartitionedCallPartitionedCall+my_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75528
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_75871conv2d_1_75873*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75540І
$my_dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0#^my_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_75558ф
flatten/PartitionedCallPartitionedCall-my_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџШL* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75566
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_75878dense_75880*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75579
$my_dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0%^my_dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_75597
dense_1/StatefulPartitionedCallStatefulPartitionedCall-my_dropout_2/StatefulPartitionedCall:output:0dense_1_75884dense_1_75886*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75609ф
activation/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_75620r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
П
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall#^my_dropout/StatefulPartitionedCall%^my_dropout_1/StatefulPartitionedCall%^my_dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"my_dropout/StatefulPartitionedCall"my_dropout/StatefulPartitionedCall2L
$my_dropout_1/StatefulPartitionedCall$my_dropout_1/StatefulPartitionedCall2L
$my_dropout_2/StatefulPartitionedCall$my_dropout_2/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameconv2d_input


f
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_76297

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџєC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџє*
dtype0*

seed**
seed2ф[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџєp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџєj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџєZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџє"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџє:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
б	
Ь
#__inference_signature_wrapper_75950
conv2d_input!
unknown:
	unknown_0:#
	unknown_1:2
	unknown_2:2
	unknown_3:
ШLє
	unknown_4:	є
	unknown_5:	є

	unknown_6:

identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *)
f$R"
 __inference__wrapped_model_75475o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameconv2d_input
б

f
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_75558

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
dtype0*

seed**
seed2ПЉА[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2:W S
/
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ь
I
-__inference_max_pooling2d_layer_call_fn_76169

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75528h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ш,
ы
E__inference_sequential_layer_call_and_return_conditional_losses_75921
conv2d_input&
conv2d_75894:
conv2d_75896:(
conv2d_1_75901:2
conv2d_1_75903:2
dense_75908:
ШLє
dense_75910:	є 
dense_1_75914:	є

dense_1_75916:

identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂ"my_dropout/StatefulPartitionedCallЂ$my_dropout_1/StatefulPartitionedCallЂ$my_dropout_2/StatefulPartitionedCallњ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_75894conv2d_75896*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_75504ћ
"my_dropout/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_my_dropout_layer_call_and_return_conditional_losses_75755ѕ
max_pooling2d/PartitionedCallPartitionedCall+my_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75528
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_75901conv2d_1_75903*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75540І
$my_dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0#^my_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_75717ф
flatten/PartitionedCallPartitionedCall-my_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџШL* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75566
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_75908dense_75910*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75579
$my_dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0%^my_dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_75678
dense_1/StatefulPartitionedCallStatefulPartitionedCall-my_dropout_2/StatefulPartitionedCall:output:0dense_1_75914dense_1_75916*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75609ф
activation/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_75620r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
П
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall#^my_dropout/StatefulPartitionedCall%^my_dropout_1/StatefulPartitionedCall%^my_dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"my_dropout/StatefulPartitionedCall"my_dropout/StatefulPartitionedCall2L
$my_dropout_1/StatefulPartitionedCall$my_dropout_1/StatefulPartitionedCall2L
$my_dropout_2/StatefulPartitionedCall$my_dropout_2/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameconv2d_input


f
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_76285

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџєC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџє*
dtype0*

seed**
seed2јї[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџєp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџєj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџєZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџє"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџє:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
Є
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75528

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
б

f
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_75717

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
dtype0*

seed**
seed2Йе[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2:W S
/
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs

e
,__inference_my_dropout_1_layer_call_fn_76208

inputs
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_75717w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ222
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Я

d
E__inference_my_dropout_layer_call_and_return_conditional_losses_76147

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed**
seed2НїЧ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџa
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


њ
A__inference_conv2d_layer_call_and_return_conditional_losses_75504

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0{
BiasAddAddV2Conv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityBiasAdd:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


њ
A__inference_conv2d_layer_call_and_return_conditional_losses_76125

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0{
BiasAddAddV2Conv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityBiasAdd:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я

(__inference_conv2d_1_layer_call_fn_76188

inputs!
unknown:2
	unknown_0:2
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75540w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы	
Э
*__inference_sequential_layer_call_fn_75971

inputs!
unknown:
	unknown_0:#
	unknown_1:2
	unknown_2:2
	unknown_3:
ШLє
	unknown_4:	є
	unknown_5:	є

	unknown_6:

identityЂStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_75623o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


f
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_75597

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџєC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ј
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџє*
dtype0*

seed**
seed2ЛЇ[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџєp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџєj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџєZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџє"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџє:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76174

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я

d
E__inference_my_dropout_layer_call_and_return_conditional_losses_75522

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed**
seed2Ь[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџa
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§	
г
*__inference_sequential_layer_call_fn_75861
conv2d_input!
unknown:
	unknown_0:#
	unknown_1:2
	unknown_2:2
	unknown_3:
ШLє
	unknown_4:	є
	unknown_5:	є

	unknown_6:

identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_75821o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameconv2d_input
Ю
a
E__inference_activation_layer_call_and_return_conditional_losses_76326

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:џџџџџџџџџ
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ
:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
Ф
^
B__inference_flatten_layer_call_and_return_conditional_losses_75566

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH&  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџШLY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџШL"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2:W S
/
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ш

'__inference_dense_1_layer_call_fn_76306

inputs
unknown:	є

	unknown_0:

identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75609o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџє: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
ЙM
Щ
 __inference__wrapped_model_75475
conv2d_inputJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:2A
3sequential_conv2d_1_biasadd_readvariableop_resource:2C
/sequential_dense_matmul_readvariableop_resource:
ШLє?
0sequential_dense_biasadd_readvariableop_resource:	єD
1sequential_dense_1_matmul_readvariableop_resource:	є
@
2sequential_dense_1_biasadd_readvariableop_resource:

identityЂ(sequential/conv2d/BiasAdd/ReadVariableOpЂ'sequential/conv2d/Conv2D/ReadVariableOpЂ*sequential/conv2d_1/BiasAdd/ReadVariableOpЂ)sequential/conv2d_1/Conv2D/ReadVariableOpЂ'sequential/dense/BiasAdd/ReadVariableOpЂ&sequential/dense/MatMul/ReadVariableOpЂ)sequential/dense_1/BiasAdd/ReadVariableOpЂ(sequential/dense_1/MatMul/ReadVariableOp 
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0У
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
sequential/conv2d/BiasAddAddV2!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџh
#sequential/my_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Џ
!sequential/my_dropout/dropout/MulMulsequential/conv2d/BiasAdd:z:0,sequential/my_dropout/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџp
#sequential/my_dropout/dropout/ShapeShapesequential/conv2d/BiasAdd:z:0*
T0*
_output_shapes
:л
:sequential/my_dropout/dropout/random_uniform/RandomUniformRandomUniform,sequential/my_dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed**
seed2ыыkq
,sequential/my_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?№
*sequential/my_dropout/dropout/GreaterEqualGreaterEqualCsequential/my_dropout/dropout/random_uniform/RandomUniform:output:05sequential/my_dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџЃ
"sequential/my_dropout/dropout/CastCast.sequential/my_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџГ
#sequential/my_dropout/dropout/Mul_1Mul%sequential/my_dropout/dropout/Mul:z:0&sequential/my_dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџС
 sequential/max_pooling2d/MaxPoolMaxPool'sequential/my_dropout/dropout/Mul_1:z:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
Є
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0ф
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
paddingSAME*
strides

*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0З
sequential/conv2d_1/BiasAddAddV2#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2j
%sequential/my_dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Е
#sequential/my_dropout_1/dropout/MulMulsequential/conv2d_1/BiasAdd:z:0.sequential/my_dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2t
%sequential/my_dropout_1/dropout/ShapeShapesequential/conv2d_1/BiasAdd:z:0*
T0*
_output_shapes
:р
<sequential/my_dropout_1/dropout/random_uniform/RandomUniformRandomUniform.sequential/my_dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
dtype0*

seed**
seed2ћs
.sequential/my_dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?і
,sequential/my_dropout_1/dropout/GreaterEqualGreaterEqualEsequential/my_dropout_1/dropout/random_uniform/RandomUniform:output:07sequential/my_dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2Ї
$sequential/my_dropout_1/dropout/CastCast0sequential/my_dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2Й
%sequential/my_dropout_1/dropout/Mul_1Mul'sequential/my_dropout_1/dropout/Mul:z:0(sequential/my_dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџH&  І
sequential/flatten/ReshapeReshape)sequential/my_dropout_1/dropout/Mul_1:z:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџШL
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
ШLє*
dtype0Љ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџє
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:є*
dtype0Ј
sequential/dense/BiasAddAddV2!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџєn
sequential/dense/ReluRelusequential/dense/BiasAdd:z:0*
T0*(
_output_shapes
:џџџџџџџџџєj
%sequential/my_dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @В
#sequential/my_dropout_2/dropout/MulMul#sequential/dense/Relu:activations:0.sequential/my_dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџєx
%sequential/my_dropout_2/dropout/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:й
<sequential/my_dropout_2/dropout/random_uniform/RandomUniformRandomUniform.sequential/my_dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџє*
dtype0*

seed**
seed2ЅЖяs
.sequential/my_dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?я
,sequential/my_dropout_2/dropout/GreaterEqualGreaterEqualEsequential/my_dropout_2/dropout/random_uniform/RandomUniform:output:07sequential/my_dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџє 
$sequential/my_dropout_2/dropout/CastCast0sequential/my_dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџєВ
%sequential/my_dropout_2/dropout/Mul_1Mul'sequential/my_dropout_2/dropout/Mul:z:0(sequential/my_dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџє
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	є
*
dtype0В
sequential/dense_1/MatMulMatMul)sequential/my_dropout_2/dropout/Mul_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0­
sequential/dense_1/BiasAddAddV2#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
sequential/activation/SoftmaxSoftmaxsequential/dense_1/BiasAdd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
v
IdentityIdentity'sequential/activation/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameconv2d_input
Т	
є
B__inference_dense_1_layer_call_and_return_conditional_losses_75609

inputs1
matmul_readvariableop_resource:	є
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	є
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0t
BiasAddAddV2MatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Z
IdentityIdentityBiasAdd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџє: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
В
C
'__inference_flatten_layer_call_fn_76237

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџШL* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75566a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџШL"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2:W S
/
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs

e
,__inference_my_dropout_2_layer_call_fn_76273

inputs
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџє* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_75678p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџє`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџє22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџє
 
_user_specified_nameinputs
§	
г
*__inference_sequential_layer_call_fn_75642
conv2d_input!
unknown:
	unknown_0:#
	unknown_1:2
	unknown_2:2
	unknown_3:
ШLє
	unknown_4:	є
	unknown_5:	є

	unknown_6:

identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*4
config_proto$"

CPU

GPU(2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_75623o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameconv2d_input
б

f
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_76220

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:А
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
dtype0*

seed**
seed2НК[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2:W S
/
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs

e
,__inference_my_dropout_1_layer_call_fn_76203

inputs
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8 *P
fKRI
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_75558w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ222
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*П
serving_defaultЋ
M
conv2d_input=
serving_default_conv2d_input:0џџџџџџџџџ>

activation0
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:фЖ
г
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
__call__
+&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_sequential
Н

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
Н

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
%	variables
&trainable_variables
'regularization_losses
(	keras_api
Ѓ__call__
+Є&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
)	variables
*trainable_variables
+regularization_losses
,	keras_api
Ѕ__call__
+І&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
Ї__call__
+Ј&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
3	variables
4trainable_variables
5regularization_losses
6	keras_api
Љ__call__
+Њ&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
Ћ__call__
+Ќ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
=	variables
>trainable_variables
?regularization_losses
@	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses"
_tf_keras_layer
ѓ
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratemmm m-m.m7m8mvvv v-v.v7v8v"
	optimizer
X
0
1
2
 3
-4
.5
76
87"
trackable_list_wrapper
X
0
1
2
 3
-4
.5
76
87"
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
Џserving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
А
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
):'22conv2d_1/kernel
:22conv2d_1/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
А
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
!	variables
"trainable_variables
#regularization_losses
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
%	variables
&trainable_variables
'regularization_losses
Ѓ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
)	variables
*trainable_variables
+regularization_losses
Ѕ__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
 :
ШLє2dense/kernel
:є2
dense/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
А
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
/	variables
0trainable_variables
1regularization_losses
Ї__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
3	variables
4trainable_variables
5regularization_losses
Љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
!:	є
2dense_1/kernel
:
2dense_1/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
А
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
9	variables
:trainable_variables
;regularization_losses
Ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
=	variables
>trainable_variables
?regularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
	total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
/
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,22Adam/conv2d_1/kernel/m
 :22Adam/conv2d_1/bias/m
%:#
ШLє2Adam/dense/kernel/m
:є2Adam/dense/bias/m
&:$	є
2Adam/dense_1/kernel/m
:
2Adam/dense_1/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,22Adam/conv2d_1/kernel/v
 :22Adam/conv2d_1/bias/v
%:#
ШLє2Adam/dense/kernel/v
:є2Adam/dense/bias/v
&:$	є
2Adam/dense_1/kernel/v
:
2Adam/dense_1/bias/v
і2ѓ
*__inference_sequential_layer_call_fn_75642
*__inference_sequential_layer_call_fn_75971
*__inference_sequential_layer_call_fn_75992
*__inference_sequential_layer_call_fn_75861Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2п
E__inference_sequential_layer_call_and_return_conditional_losses_76049
E__inference_sequential_layer_call_and_return_conditional_losses_76106
E__inference_sequential_layer_call_and_return_conditional_losses_75891
E__inference_sequential_layer_call_and_return_conditional_losses_75921Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
аBЭ
 __inference__wrapped_model_75475conv2d_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv2d_layer_call_fn_76115Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_conv2d_layer_call_and_return_conditional_losses_76125Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
*__inference_my_dropout_layer_call_fn_76130
*__inference_my_dropout_layer_call_fn_76135Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ш2Х
E__inference_my_dropout_layer_call_and_return_conditional_losses_76147
E__inference_my_dropout_layer_call_and_return_conditional_losses_76159Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
-__inference_max_pooling2d_layer_call_fn_76164
-__inference_max_pooling2d_layer_call_fn_76169Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
М2Й
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76174
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76179Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_conv2d_1_layer_call_fn_76188Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_76198Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
,__inference_my_dropout_1_layer_call_fn_76203
,__inference_my_dropout_1_layer_call_fn_76208Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2Щ
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_76220
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_76232Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
б2Ю
'__inference_flatten_layer_call_fn_76237Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_flatten_layer_call_and_return_conditional_losses_76243Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
%__inference_dense_layer_call_fn_76252Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъ2ч
@__inference_dense_layer_call_and_return_conditional_losses_76263Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
,__inference_my_dropout_2_layer_call_fn_76268
,__inference_my_dropout_2_layer_call_fn_76273Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2Щ
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_76285
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_76297Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
б2Ю
'__inference_dense_1_layer_call_fn_76306Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_1_layer_call_and_return_conditional_losses_76316Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_activation_layer_call_fn_76321Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_activation_layer_call_and_return_conditional_losses_76326Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЯBЬ
#__inference_signature_wrapper_75950conv2d_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 Ї
 __inference__wrapped_model_75475 -.78=Ђ:
3Ђ0
.+
conv2d_inputџџџџџџџџџ
Њ "7Њ4
2

activation$!

activationџџџџџџџџџ
Ё
E__inference_activation_layer_call_and_return_conditional_losses_76326X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ "%Ђ"

0џџџџџџџџџ

 y
*__inference_activation_layer_call_fn_76321K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ "џџџџџџџџџ
Г
C__inference_conv2d_1_layer_call_and_return_conditional_losses_76198l 7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ2
 
(__inference_conv2d_1_layer_call_fn_76188_ 7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ2Б
A__inference_conv2d_layer_call_and_return_conditional_losses_76125l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
&__inference_conv2d_layer_call_fn_76115_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЃ
B__inference_dense_1_layer_call_and_return_conditional_losses_76316]780Ђ-
&Ђ#
!
inputsџџџџџџџџџє
Њ "%Ђ"

0џџџџџџџџџ

 {
'__inference_dense_1_layer_call_fn_76306P780Ђ-
&Ђ#
!
inputsџџџџџџџџџє
Њ "џџџџџџџџџ
Ђ
@__inference_dense_layer_call_and_return_conditional_losses_76263^-.0Ђ-
&Ђ#
!
inputsџџџџџџџџџШL
Њ "&Ђ#

0џџџџџџџџџє
 z
%__inference_dense_layer_call_fn_76252Q-.0Ђ-
&Ђ#
!
inputsџџџџџџџџџШL
Њ "џџџџџџџџџєЇ
B__inference_flatten_layer_call_and_return_conditional_losses_76243a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ2
Њ "&Ђ#

0џџџџџџџџџШL
 
'__inference_flatten_layer_call_fn_76237T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ2
Њ "џџџџџџџџџШLы
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76174RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Д
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76179h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 У
-__inference_max_pooling2d_layer_call_fn_76164RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
-__inference_max_pooling2d_layer_call_fn_76169[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЗ
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_76220l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ2
p 
Њ "-Ђ*
# 
0џџџџџџџџџ2
 З
G__inference_my_dropout_1_layer_call_and_return_conditional_losses_76232l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ2
p
Њ "-Ђ*
# 
0џџџџџџџџџ2
 
,__inference_my_dropout_1_layer_call_fn_76203_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ2
p 
Њ " џџџџџџџџџ2
,__inference_my_dropout_1_layer_call_fn_76208_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ2
p
Њ " џџџџџџџџџ2Љ
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_76285^4Ђ1
*Ђ'
!
inputsџџџџџџџџџє
p 
Њ "&Ђ#

0џџџџџџџџџє
 Љ
G__inference_my_dropout_2_layer_call_and_return_conditional_losses_76297^4Ђ1
*Ђ'
!
inputsџџџџџџџџџє
p
Њ "&Ђ#

0џџџџџџџџџє
 
,__inference_my_dropout_2_layer_call_fn_76268Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџє
p 
Њ "џџџџџџџџџє
,__inference_my_dropout_2_layer_call_fn_76273Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџє
p
Њ "џџџџџџџџџєЕ
E__inference_my_dropout_layer_call_and_return_conditional_losses_76147l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 Е
E__inference_my_dropout_layer_call_and_return_conditional_losses_76159l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 
*__inference_my_dropout_layer_call_fn_76130_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ " џџџџџџџџџ
*__inference_my_dropout_layer_call_fn_76135_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ " џџџџџџџџџС
E__inference_sequential_layer_call_and_return_conditional_losses_75891x -.78EЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 С
E__inference_sequential_layer_call_and_return_conditional_losses_75921x -.78EЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 Л
E__inference_sequential_layer_call_and_return_conditional_losses_76049r -.78?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Л
E__inference_sequential_layer_call_and_return_conditional_losses_76106r -.78?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 
*__inference_sequential_layer_call_fn_75642k -.78EЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

*__inference_sequential_layer_call_fn_75861k -.78EЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

*__inference_sequential_layer_call_fn_75971e -.78?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

*__inference_sequential_layer_call_fn_75992e -.78?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
К
#__inference_signature_wrapper_75950 -.78MЂJ
Ђ 
CЊ@
>
conv2d_input.+
conv2d_inputџџџџџџџџџ"7Њ4
2

activation$!

activationџџџџџџџџџ
