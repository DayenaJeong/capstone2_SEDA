НЉ/
лЌ
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
О
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8-
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:  *
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
: *
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:  *
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
: *
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

: *
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
\
iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameiter
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0

lstm_5/lstm_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	&*+
shared_namelstm_5/lstm_cell_10/kernel

.lstm_5/lstm_cell_10/kernel/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/kernel*
_output_shapes
:	&*
dtype0
Ѕ
$lstm_5/lstm_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *5
shared_name&$lstm_5/lstm_cell_10/recurrent_kernel

8lstm_5/lstm_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp$lstm_5/lstm_cell_10/recurrent_kernel*
_output_shapes
:	 *
dtype0

lstm_5/lstm_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namelstm_5/lstm_cell_10/bias

,lstm_5/lstm_cell_10/bias/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/bias*
_output_shapes	
:*
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
~
dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *"
shared_namedense_15/kernel/m
w
%dense_15/kernel/m/Read/ReadVariableOpReadVariableOpdense_15/kernel/m*
_output_shapes

:  *
dtype0
v
dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namedense_15/bias/m
o
#dense_15/bias/m/Read/ReadVariableOpReadVariableOpdense_15/bias/m*
_output_shapes
: *
dtype0
~
dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *"
shared_namedense_16/kernel/m
w
%dense_16/kernel/m/Read/ReadVariableOpReadVariableOpdense_16/kernel/m*
_output_shapes

:  *
dtype0
v
dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namedense_16/bias/m
o
#dense_16/bias/m/Read/ReadVariableOpReadVariableOpdense_16/bias/m*
_output_shapes
: *
dtype0
~
dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namedense_17/kernel/m
w
%dense_17/kernel/m/Read/ReadVariableOpReadVariableOpdense_17/kernel/m*
_output_shapes

: *
dtype0
v
dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_17/bias/m
o
#dense_17/bias/m/Read/ReadVariableOpReadVariableOpdense_17/bias/m*
_output_shapes
:*
dtype0

lstm_5/lstm_cell_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	&*-
shared_namelstm_5/lstm_cell_10/kernel/m

0lstm_5/lstm_cell_10/kernel/m/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/kernel/m*
_output_shapes
:	&*
dtype0
Љ
&lstm_5/lstm_cell_10/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&lstm_5/lstm_cell_10/recurrent_kernel/m
Ђ
:lstm_5/lstm_cell_10/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&lstm_5/lstm_cell_10/recurrent_kernel/m*
_output_shapes
:	 *
dtype0

lstm_5/lstm_cell_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelstm_5/lstm_cell_10/bias/m

.lstm_5/lstm_cell_10/bias/m/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/bias/m*
_output_shapes	
:*
dtype0
~
dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *"
shared_namedense_15/kernel/v
w
%dense_15/kernel/v/Read/ReadVariableOpReadVariableOpdense_15/kernel/v*
_output_shapes

:  *
dtype0
v
dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namedense_15/bias/v
o
#dense_15/bias/v/Read/ReadVariableOpReadVariableOpdense_15/bias/v*
_output_shapes
: *
dtype0
~
dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *"
shared_namedense_16/kernel/v
w
%dense_16/kernel/v/Read/ReadVariableOpReadVariableOpdense_16/kernel/v*
_output_shapes

:  *
dtype0
v
dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namedense_16/bias/v
o
#dense_16/bias/v/Read/ReadVariableOpReadVariableOpdense_16/bias/v*
_output_shapes
: *
dtype0
~
dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namedense_17/kernel/v
w
%dense_17/kernel/v/Read/ReadVariableOpReadVariableOpdense_17/kernel/v*
_output_shapes

: *
dtype0
v
dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_17/bias/v
o
#dense_17/bias/v/Read/ReadVariableOpReadVariableOpdense_17/bias/v*
_output_shapes
:*
dtype0

lstm_5/lstm_cell_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	&*-
shared_namelstm_5/lstm_cell_10/kernel/v

0lstm_5/lstm_cell_10/kernel/v/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/kernel/v*
_output_shapes
:	&*
dtype0
Љ
&lstm_5/lstm_cell_10/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&lstm_5/lstm_cell_10/recurrent_kernel/v
Ђ
:lstm_5/lstm_cell_10/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&lstm_5/lstm_cell_10/recurrent_kernel/v*
_output_shapes
:	 *
dtype0

lstm_5/lstm_cell_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelstm_5/lstm_cell_10/bias/v

.lstm_5/lstm_cell_10/bias/v/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
К5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ѕ4
valueы4Bш4 Bс4

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
h

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
т
(iter

)beta_1

*beta_2
	+decay
,learning_ratemcmdmemf"mg#mh-mi.mj/mkvlvmvnvo"vp#vq-vr.vs/vt
?
-0
.1
/2
3
4
5
6
"7
#8
 
?
-0
.1
/2
3
4
5
6
"7
#8
­
0metrics
	variables

1layers
2layer_regularization_losses
3non_trainable_variables
4layer_metrics
regularization_losses
	trainable_variables
 
~

-kernel
.recurrent_kernel
/bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
 

-0
.1
/2
 

-0
.1
/2
Й
9metrics
	variables

:layers
;layer_regularization_losses

<states
=non_trainable_variables
>layer_metrics
regularization_losses
trainable_variables
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
?metrics
	variables

@layers
Alayer_regularization_losses
Bnon_trainable_variables
Clayer_metrics
regularization_losses
trainable_variables
 
 
 
­
Dmetrics
	variables

Elayers
Flayer_regularization_losses
Gnon_trainable_variables
Hlayer_metrics
regularization_losses
trainable_variables
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
Imetrics
	variables

Jlayers
Klayer_regularization_losses
Lnon_trainable_variables
Mlayer_metrics
regularization_losses
 trainable_variables
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_17/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
­
Nmetrics
$	variables

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rlayer_metrics
%regularization_losses
&trainable_variables
CA
VARIABLE_VALUEiter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElstm_5/lstm_cell_10/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$lstm_5/lstm_cell_10/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUElstm_5/lstm_cell_10/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE

S0
T1
#
0
1
2
3
4
 
 
 

-0
.1
/2
 

-0
.1
/2
­
Umetrics
5	variables

Vlayers
Wlayer_regularization_losses
Xnon_trainable_variables
Ylayer_metrics
6regularization_losses
7trainable_variables
 

0
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
4
	Ztotal
	[count
\	variables
]	keras_api
D
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1

\	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

^0
_1

a	variables
yw
VARIABLE_VALUEdense_15/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_15/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_16/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_16/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_17/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_17/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUElstm_5/lstm_cell_10/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&lstm_5/lstm_cell_10/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUElstm_5/lstm_cell_10/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_15/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_15/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_16/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_16/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_17/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_17/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUElstm_5/lstm_cell_10/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&lstm_5/lstm_cell_10/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUElstm_5/lstm_cell_10/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lstm_5_inputPlaceholder*+
_output_shapes
:џџџџџџџџџ&*
dtype0* 
shape:џџџџџџџџџ&

StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_5_inputlstm_5/lstm_cell_10/kernellstm_5/lstm_cell_10/bias$lstm_5/lstm_cell_10/recurrent_kerneldense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_188679
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Г
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpiter/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp.lstm_5/lstm_cell_10/kernel/Read/ReadVariableOp8lstm_5/lstm_cell_10/recurrent_kernel/Read/ReadVariableOp,lstm_5/lstm_cell_10/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp%dense_15/kernel/m/Read/ReadVariableOp#dense_15/bias/m/Read/ReadVariableOp%dense_16/kernel/m/Read/ReadVariableOp#dense_16/bias/m/Read/ReadVariableOp%dense_17/kernel/m/Read/ReadVariableOp#dense_17/bias/m/Read/ReadVariableOp0lstm_5/lstm_cell_10/kernel/m/Read/ReadVariableOp:lstm_5/lstm_cell_10/recurrent_kernel/m/Read/ReadVariableOp.lstm_5/lstm_cell_10/bias/m/Read/ReadVariableOp%dense_15/kernel/v/Read/ReadVariableOp#dense_15/bias/v/Read/ReadVariableOp%dense_16/kernel/v/Read/ReadVariableOp#dense_16/bias/v/Read/ReadVariableOp%dense_17/kernel/v/Read/ReadVariableOp#dense_17/bias/v/Read/ReadVariableOp0lstm_5/lstm_cell_10/kernel/v/Read/ReadVariableOp:lstm_5/lstm_cell_10/recurrent_kernel/v/Read/ReadVariableOp.lstm_5/lstm_cell_10/bias/v/Read/ReadVariableOpConst*1
Tin*
(2&	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_191360
о
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasiterbeta_1beta_2decaylearning_ratelstm_5/lstm_cell_10/kernel$lstm_5/lstm_cell_10/recurrent_kernellstm_5/lstm_cell_10/biastotalcounttotal_1count_1dense_15/kernel/mdense_15/bias/mdense_16/kernel/mdense_16/bias/mdense_17/kernel/mdense_17/bias/mlstm_5/lstm_cell_10/kernel/m&lstm_5/lstm_cell_10/recurrent_kernel/mlstm_5/lstm_cell_10/bias/mdense_15/kernel/vdense_15/bias/vdense_16/kernel/vdense_16/bias/vdense_17/kernel/vdense_17/bias/vlstm_5/lstm_cell_10/kernel/v&lstm_5/lstm_cell_10/recurrent_kernel/vlstm_5/lstm_cell_10/bias/v*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_191478Є,
ѓј
в
while_body_189633
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resourceЂ!while/lstm_cell_10/ReadVariableOpЂ#while/lstm_cell_10/ReadVariableOp_1Ђ#while/lstm_cell_10/ReadVariableOp_2Ђ#while/lstm_cell_10/ReadVariableOp_3Ђ'while/lstm_cell_10/split/ReadVariableOpЂ)while/lstm_cell_10/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЈ
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_10/ones_like/Constа
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/ones_like
 while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 while/lstm_cell_10/dropout/ConstЫ
while/lstm_cell_10/dropout/MulMul%while/lstm_cell_10/ones_like:output:0)while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2 
while/lstm_cell_10/dropout/Mul
 while/lstm_cell_10/dropout/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_10/dropout/Shape
7while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ТВж29
7while/lstm_cell_10/dropout/random_uniform/RandomUniform
)while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2+
)while/lstm_cell_10/dropout/GreaterEqual/y
'while/lstm_cell_10/dropout/GreaterEqualGreaterEqual@while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2)
'while/lstm_cell_10/dropout/GreaterEqualИ
while/lstm_cell_10/dropout/CastCast+while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2!
while/lstm_cell_10/dropout/CastЦ
 while/lstm_cell_10/dropout/Mul_1Mul"while/lstm_cell_10/dropout/Mul:z:0#while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout/Mul_1
"while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_1/Constб
 while/lstm_cell_10/dropout_1/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout_1/Mul
"while/lstm_cell_10/dropout_1/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_1/Shape
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ѕї2;
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_1/GreaterEqual/y
)while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)while/lstm_cell_10/dropout_1/GreaterEqualО
!while/lstm_cell_10/dropout_1/CastCast-while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2#
!while/lstm_cell_10/dropout_1/CastЮ
"while/lstm_cell_10/dropout_1/Mul_1Mul$while/lstm_cell_10/dropout_1/Mul:z:0%while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2$
"while/lstm_cell_10/dropout_1/Mul_1
"while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_2/Constб
 while/lstm_cell_10/dropout_2/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout_2/Mul
"while/lstm_cell_10/dropout_2/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_2/Shape
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ь2;
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_2/GreaterEqual/y
)while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)while/lstm_cell_10/dropout_2/GreaterEqualО
!while/lstm_cell_10/dropout_2/CastCast-while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2#
!while/lstm_cell_10/dropout_2/CastЮ
"while/lstm_cell_10/dropout_2/Mul_1Mul$while/lstm_cell_10/dropout_2/Mul:z:0%while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2$
"while/lstm_cell_10/dropout_2/Mul_1
"while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_3/Constб
 while/lstm_cell_10/dropout_3/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout_3/Mul
"while/lstm_cell_10/dropout_3/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_3/Shape
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2цЛ2;
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_3/GreaterEqual/y
)while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)while/lstm_cell_10/dropout_3/GreaterEqualО
!while/lstm_cell_10/dropout_3/CastCast-while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2#
!while/lstm_cell_10/dropout_3/CastЮ
"while/lstm_cell_10/dropout_3/Mul_1Mul$while/lstm_cell_10/dropout_3/Mul:z:0%while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2$
"while/lstm_cell_10/dropout_3/Mul_1
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$while/lstm_cell_10/ones_like_1/Constи
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_10/ones_like_1
"while/lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_4/Constг
 while/lstm_cell_10/dropout_4/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_4/Mul
"while/lstm_cell_10/dropout_4/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_4/Shape
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Єѓ2;
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_4/GreaterEqual/y
)while/lstm_cell_10/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_4/GreaterEqualО
!while/lstm_cell_10/dropout_4/CastCast-while/lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_4/CastЮ
"while/lstm_cell_10/dropout_4/Mul_1Mul$while/lstm_cell_10/dropout_4/Mul:z:0%while/lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_4/Mul_1
"while/lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_5/Constг
 while/lstm_cell_10/dropout_5/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_5/Mul
"while/lstm_cell_10/dropout_5/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_5/Shape
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ОЧЅ2;
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_5/GreaterEqual/y
)while/lstm_cell_10/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_5/GreaterEqualО
!while/lstm_cell_10/dropout_5/CastCast-while/lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_5/CastЮ
"while/lstm_cell_10/dropout_5/Mul_1Mul$while/lstm_cell_10/dropout_5/Mul:z:0%while/lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_5/Mul_1
"while/lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_6/Constг
 while/lstm_cell_10/dropout_6/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_6/Mul
"while/lstm_cell_10/dropout_6/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_6/Shape
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЫЊ2;
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_6/GreaterEqual/y
)while/lstm_cell_10/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_6/GreaterEqualО
!while/lstm_cell_10/dropout_6/CastCast-while/lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_6/CastЮ
"while/lstm_cell_10/dropout_6/Mul_1Mul$while/lstm_cell_10/dropout_6/Mul:z:0%while/lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_6/Mul_1
"while/lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_7/Constг
 while/lstm_cell_10/dropout_7/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_7/Mul
"while/lstm_cell_10/dropout_7/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_7/Shape
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2хєЌ2;
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_7/GreaterEqual/y
)while/lstm_cell_10/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_7/GreaterEqualО
!while/lstm_cell_10/dropout_7/CastCast-while/lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_7/CastЮ
"while/lstm_cell_10/dropout_7/Mul_1Mul$while/lstm_cell_10/dropout_7/Mul:z:0%while/lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_7/Mul_1С
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mulЧ
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_1Ч
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_2Ч
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЦ
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpѓ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/splitБ
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMulЗ
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_1З
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_2З
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dimШ
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpы
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1П
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAddХ
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_1Х
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_2Х
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_3Њ
while/lstm_cell_10/mul_4Mulwhile_placeholder_2&while/lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_4Њ
while/lstm_cell_10/mul_5Mulwhile_placeholder_2&while/lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_5Њ
while/lstm_cell_10/mul_6Mulwhile_placeholder_2&while/lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_6Њ
while/lstm_cell_10/mul_7Mulwhile_placeholder_2&while/lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_7Д
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_10/ReadVariableOpЁ
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stackЅ
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1Ѕ
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2ю
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceП
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_4З
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/SigmoidИ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_1Ѕ
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stackЉ
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1Љ
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2њ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1С
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_5Н
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_1Є
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_8И
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_2Ѕ
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stackЉ
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1Љ
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2њ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2С
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_6Н
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_2
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/TanhЊ
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_9Ћ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_3И
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_3Ѕ
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stackЉ
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1Љ
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2њ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3С
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_7Н
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_4
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Tanh_1А
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_10с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3э
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4ь
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ :џџџџџџџџџ : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
Ќ
Э
"__inference__traced_restore_191478
file_prefix$
 assignvariableop_dense_15_kernel$
 assignvariableop_1_dense_15_bias&
"assignvariableop_2_dense_16_kernel$
 assignvariableop_3_dense_16_bias&
"assignvariableop_4_dense_17_kernel$
 assignvariableop_5_dense_17_bias
assignvariableop_6_iter
assignvariableop_7_beta_1
assignvariableop_8_beta_2
assignvariableop_9_decay%
!assignvariableop_10_learning_rate2
.assignvariableop_11_lstm_5_lstm_cell_10_kernel<
8assignvariableop_12_lstm_5_lstm_cell_10_recurrent_kernel0
,assignvariableop_13_lstm_5_lstm_cell_10_bias
assignvariableop_14_total
assignvariableop_15_count
assignvariableop_16_total_1
assignvariableop_17_count_1)
%assignvariableop_18_dense_15_kernel_m'
#assignvariableop_19_dense_15_bias_m)
%assignvariableop_20_dense_16_kernel_m'
#assignvariableop_21_dense_16_bias_m)
%assignvariableop_22_dense_17_kernel_m'
#assignvariableop_23_dense_17_bias_m4
0assignvariableop_24_lstm_5_lstm_cell_10_kernel_m>
:assignvariableop_25_lstm_5_lstm_cell_10_recurrent_kernel_m2
.assignvariableop_26_lstm_5_lstm_cell_10_bias_m)
%assignvariableop_27_dense_15_kernel_v'
#assignvariableop_28_dense_15_bias_v)
%assignvariableop_29_dense_16_kernel_v'
#assignvariableop_30_dense_16_bias_v)
%assignvariableop_31_dense_17_kernel_v'
#assignvariableop_32_dense_17_bias_v4
0assignvariableop_33_lstm_5_lstm_cell_10_kernel_v>
:assignvariableop_34_lstm_5_lstm_cell_10_recurrent_kernel_v2
.assignvariableop_35_lstm_5_lstm_cell_10_bias_v
identity_37ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*Ў
valueЄBЁ%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesи
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesч
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Њ
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_16_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_16_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ї
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_17_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_17_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Љ
AssignVariableOp_10AssignVariableOp!assignvariableop_10_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ж
AssignVariableOp_11AssignVariableOp.assignvariableop_11_lstm_5_lstm_cell_10_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Р
AssignVariableOp_12AssignVariableOp8assignvariableop_12_lstm_5_lstm_cell_10_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Д
AssignVariableOp_13AssignVariableOp,assignvariableop_13_lstm_5_lstm_cell_10_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ё
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ё
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ѓ
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ѓ
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18­
AssignVariableOp_18AssignVariableOp%assignvariableop_18_dense_15_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ћ
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_15_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20­
AssignVariableOp_20AssignVariableOp%assignvariableop_20_dense_16_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ћ
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_16_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22­
AssignVariableOp_22AssignVariableOp%assignvariableop_22_dense_17_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ћ
AssignVariableOp_23AssignVariableOp#assignvariableop_23_dense_17_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24И
AssignVariableOp_24AssignVariableOp0assignvariableop_24_lstm_5_lstm_cell_10_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Т
AssignVariableOp_25AssignVariableOp:assignvariableop_25_lstm_5_lstm_cell_10_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ж
AssignVariableOp_26AssignVariableOp.assignvariableop_26_lstm_5_lstm_cell_10_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27­
AssignVariableOp_27AssignVariableOp%assignvariableop_27_dense_15_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ћ
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_15_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29­
AssignVariableOp_29AssignVariableOp%assignvariableop_29_dense_16_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ћ
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_16_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31­
AssignVariableOp_31AssignVariableOp%assignvariableop_31_dense_17_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ћ
AssignVariableOp_32AssignVariableOp#assignvariableop_32_dense_17_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33И
AssignVariableOp_33AssignVariableOp0assignvariableop_33_lstm_5_lstm_cell_10_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Т
AssignVariableOp_34AssignVariableOp:assignvariableop_34_lstm_5_lstm_cell_10_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ж
AssignVariableOp_35AssignVariableOp.assignvariableop_35_lstm_5_lstm_cell_10_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_359
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpі
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_36щ
Identity_37IdentityIdentity_36:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_37"#
identity_37Identity_37:output:0*Ї
_input_shapes
: ::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
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
 щ
п
H__inference_sequential_5_layer_call_and_return_conditional_losses_189392

inputs5
1lstm_5_lstm_cell_10_split_readvariableop_resource7
3lstm_5_lstm_cell_10_split_1_readvariableop_resource/
+lstm_5_lstm_cell_10_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identityЂdense_15/BiasAdd/ReadVariableOpЂdense_15/MatMul/ReadVariableOpЂdense_16/BiasAdd/ReadVariableOpЂdense_16/MatMul/ReadVariableOpЂdense_17/BiasAdd/ReadVariableOpЂdense_17/MatMul/ReadVariableOpЂ"lstm_5/lstm_cell_10/ReadVariableOpЂ$lstm_5/lstm_cell_10/ReadVariableOp_1Ђ$lstm_5/lstm_cell_10/ReadVariableOp_2Ђ$lstm_5/lstm_cell_10/ReadVariableOp_3Ђ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂ(lstm_5/lstm_cell_10/split/ReadVariableOpЂ*lstm_5/lstm_cell_10/split_1/ReadVariableOpЂlstm_5/whileR
lstm_5/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_5/Shape
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros/mul/y
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_5/zeros/Less/y
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros/packed/1
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros_1/mul/y
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_5/zeros_1/Less/y
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros_1/packed/1Ѕ
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/zeros_1
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose/perm
lstm_5/transpose	Transposeinputslstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ&2
lstm_5/transposed
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:2
lstm_5/Shape_1
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_1/stack
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_1
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_2
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slice_1
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"lstm_5/TensorArrayV2/element_shapeЮ
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2Э
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   2>
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_5/TensorArrayUnstack/TensorListFromTensor
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_2/stack
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_1
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_2І
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ&*
shrink_axis_mask2
lstm_5/strided_slice_2
#lstm_5/lstm_cell_10/ones_like/ShapeShapelstm_5/strided_slice_2:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/ones_like/Shape
#lstm_5/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_5/lstm_cell_10/ones_like/Constд
lstm_5/lstm_cell_10/ones_likeFill,lstm_5/lstm_cell_10/ones_like/Shape:output:0,lstm_5/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/lstm_cell_10/ones_like
%lstm_5/lstm_cell_10/ones_like_1/ShapeShapelstm_5/zeros:output:0*
T0*
_output_shapes
:2'
%lstm_5/lstm_cell_10/ones_like_1/Shape
%lstm_5/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%lstm_5/lstm_cell_10/ones_like_1/Constм
lstm_5/lstm_cell_10/ones_like_1Fill.lstm_5/lstm_cell_10/ones_like_1/Shape:output:0.lstm_5/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/lstm_cell_10/ones_like_1Д
lstm_5/lstm_cell_10/mulMullstm_5/strided_slice_2:output:0&lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/lstm_cell_10/mulИ
lstm_5/lstm_cell_10/mul_1Mullstm_5/strided_slice_2:output:0&lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/lstm_cell_10/mul_1И
lstm_5/lstm_cell_10/mul_2Mullstm_5/strided_slice_2:output:0&lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/lstm_cell_10/mul_2И
lstm_5/lstm_cell_10/mul_3Mullstm_5/strided_slice_2:output:0&lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/lstm_cell_10/mul_3x
lstm_5/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_10/Const
#lstm_5/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_5/lstm_cell_10/split/split_dimЧ
(lstm_5/lstm_cell_10/split/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02*
(lstm_5/lstm_cell_10/split/ReadVariableOpї
lstm_5/lstm_cell_10/splitSplit,lstm_5/lstm_cell_10/split/split_dim:output:00lstm_5/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_5/lstm_cell_10/splitЕ
lstm_5/lstm_cell_10/MatMulMatMullstm_5/lstm_cell_10/mul:z:0"lstm_5/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMulЛ
lstm_5/lstm_cell_10/MatMul_1MatMullstm_5/lstm_cell_10/mul_1:z:0"lstm_5/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_1Л
lstm_5/lstm_cell_10/MatMul_2MatMullstm_5/lstm_cell_10/mul_2:z:0"lstm_5/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_2Л
lstm_5/lstm_cell_10/MatMul_3MatMullstm_5/lstm_cell_10/mul_3:z:0"lstm_5/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_3|
lstm_5/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_10/Const_1
%lstm_5/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%lstm_5/lstm_cell_10/split_1/split_dimЩ
*lstm_5/lstm_cell_10/split_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02,
*lstm_5/lstm_cell_10/split_1/ReadVariableOpя
lstm_5/lstm_cell_10/split_1Split.lstm_5/lstm_cell_10/split_1/split_dim:output:02lstm_5/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_5/lstm_cell_10/split_1У
lstm_5/lstm_cell_10/BiasAddBiasAdd$lstm_5/lstm_cell_10/MatMul:product:0$lstm_5/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/BiasAddЩ
lstm_5/lstm_cell_10/BiasAdd_1BiasAdd&lstm_5/lstm_cell_10/MatMul_1:product:0$lstm_5/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/BiasAdd_1Щ
lstm_5/lstm_cell_10/BiasAdd_2BiasAdd&lstm_5/lstm_cell_10/MatMul_2:product:0$lstm_5/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/BiasAdd_2Щ
lstm_5/lstm_cell_10/BiasAdd_3BiasAdd&lstm_5/lstm_cell_10/MatMul_3:product:0$lstm_5/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/BiasAdd_3А
lstm_5/lstm_cell_10/mul_4Mullstm_5/zeros:output:0(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_4А
lstm_5/lstm_cell_10/mul_5Mullstm_5/zeros:output:0(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_5А
lstm_5/lstm_cell_10/mul_6Mullstm_5/zeros:output:0(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_6А
lstm_5/lstm_cell_10/mul_7Mullstm_5/zeros:output:0(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_7Е
"lstm_5/lstm_cell_10/ReadVariableOpReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"lstm_5/lstm_cell_10/ReadVariableOpЃ
'lstm_5/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'lstm_5/lstm_cell_10/strided_slice/stackЇ
)lstm_5/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_5/lstm_cell_10/strided_slice/stack_1Ї
)lstm_5/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)lstm_5/lstm_cell_10/strided_slice/stack_2є
!lstm_5/lstm_cell_10/strided_sliceStridedSlice*lstm_5/lstm_cell_10/ReadVariableOp:value:00lstm_5/lstm_cell_10/strided_slice/stack:output:02lstm_5/lstm_cell_10/strided_slice/stack_1:output:02lstm_5/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!lstm_5/lstm_cell_10/strided_sliceУ
lstm_5/lstm_cell_10/MatMul_4MatMullstm_5/lstm_cell_10/mul_4:z:0*lstm_5/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_4Л
lstm_5/lstm_cell_10/addAddV2$lstm_5/lstm_cell_10/BiasAdd:output:0&lstm_5/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/add
lstm_5/lstm_cell_10/SigmoidSigmoidlstm_5/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/SigmoidЙ
$lstm_5/lstm_cell_10/ReadVariableOp_1ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_1Ї
)lstm_5/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_5/lstm_cell_10/strided_slice_1/stackЋ
+lstm_5/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+lstm_5/lstm_cell_10/strided_slice_1/stack_1Ћ
+lstm_5/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_1/stack_2
#lstm_5/lstm_cell_10/strided_slice_1StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_1:value:02lstm_5/lstm_cell_10/strided_slice_1/stack:output:04lstm_5/lstm_cell_10/strided_slice_1/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_1Х
lstm_5/lstm_cell_10/MatMul_5MatMullstm_5/lstm_cell_10/mul_5:z:0,lstm_5/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_5С
lstm_5/lstm_cell_10/add_1AddV2&lstm_5/lstm_cell_10/BiasAdd_1:output:0&lstm_5/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/add_1
lstm_5/lstm_cell_10/Sigmoid_1Sigmoidlstm_5/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/Sigmoid_1Ћ
lstm_5/lstm_cell_10/mul_8Mul!lstm_5/lstm_cell_10/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_8Й
$lstm_5/lstm_cell_10/ReadVariableOp_2ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_2Ї
)lstm_5/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)lstm_5/lstm_cell_10/strided_slice_2/stackЋ
+lstm_5/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+lstm_5/lstm_cell_10/strided_slice_2/stack_1Ћ
+lstm_5/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_2/stack_2
#lstm_5/lstm_cell_10/strided_slice_2StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_2:value:02lstm_5/lstm_cell_10/strided_slice_2/stack:output:04lstm_5/lstm_cell_10/strided_slice_2/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_2Х
lstm_5/lstm_cell_10/MatMul_6MatMullstm_5/lstm_cell_10/mul_6:z:0,lstm_5/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_6С
lstm_5/lstm_cell_10/add_2AddV2&lstm_5/lstm_cell_10/BiasAdd_2:output:0&lstm_5/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/add_2
lstm_5/lstm_cell_10/TanhTanhlstm_5/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/TanhЎ
lstm_5/lstm_cell_10/mul_9Mullstm_5/lstm_cell_10/Sigmoid:y:0lstm_5/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_9Џ
lstm_5/lstm_cell_10/add_3AddV2lstm_5/lstm_cell_10/mul_8:z:0lstm_5/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/add_3Й
$lstm_5/lstm_cell_10/ReadVariableOp_3ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_3Ї
)lstm_5/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)lstm_5/lstm_cell_10/strided_slice_3/stackЋ
+lstm_5/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_5/lstm_cell_10/strided_slice_3/stack_1Ћ
+lstm_5/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_3/stack_2
#lstm_5/lstm_cell_10/strided_slice_3StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_3:value:02lstm_5/lstm_cell_10/strided_slice_3/stack:output:04lstm_5/lstm_cell_10/strided_slice_3/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_3Х
lstm_5/lstm_cell_10/MatMul_7MatMullstm_5/lstm_cell_10/mul_7:z:0,lstm_5/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_7С
lstm_5/lstm_cell_10/add_4AddV2&lstm_5/lstm_cell_10/BiasAdd_3:output:0&lstm_5/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/add_4
lstm_5/lstm_cell_10/Sigmoid_2Sigmoidlstm_5/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/Sigmoid_2
lstm_5/lstm_cell_10/Tanh_1Tanhlstm_5/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/Tanh_1Д
lstm_5/lstm_cell_10/mul_10Mul!lstm_5/lstm_cell_10/Sigmoid_2:y:0lstm_5/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_10
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2&
$lstm_5/TensorArrayV2_1/element_shapeд
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2_1\
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/time
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
lstm_5/while/maximum_iterationsx
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/while/loop_counterЭ
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_10_split_readvariableop_resource3lstm_5_lstm_cell_10_split_1_readvariableop_resource+lstm_5_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_5_while_body_189222*$
condR
lstm_5_while_cond_189221*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_5/whileУ
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    29
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02+
)lstm_5/TensorArrayV2Stack/TensorListStack
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_5/strided_slice_3/stack
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_5/strided_slice_3/stack_1
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_3/stack_2Ф
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_5/strided_slice_3
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose_1/permС
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_5/transpose_1t
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/runtimeЈ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_15/MatMul/ReadVariableOpЇ
dense_15/MatMulMatMullstm_5/strided_slice_3:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_15/MatMulЇ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_15/BiasAdd/ReadVariableOpЅ
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_15/BiasAdd|
dense_15/SigmoidSigmoiddense_15/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_15/Sigmoid|
dropout_5/IdentityIdentitydense_15/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/IdentityЈ
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_16/MatMul/ReadVariableOpЃ
dense_16/MatMulMatMuldropout_5/Identity:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_16/MatMulЇ
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_16/BiasAdd/ReadVariableOpЅ
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_16/BiasAdd|
dense_16/SigmoidSigmoiddense_16/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_16/SigmoidЈ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_17/MatMul/ReadVariableOp
dense_17/MatMulMatMuldense_16/Sigmoid:y:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_17/MatMulЇ
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOpЅ
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_17/Softmaxя
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulщ
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulД
IdentityIdentitydense_17/Softmax:softmax:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp#^lstm_5/lstm_cell_10/ReadVariableOp%^lstm_5/lstm_cell_10/ReadVariableOp_1%^lstm_5/lstm_cell_10/ReadVariableOp_2%^lstm_5/lstm_cell_10/ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp)^lstm_5/lstm_cell_10/split/ReadVariableOp+^lstm_5/lstm_cell_10/split_1/ReadVariableOp^lstm_5/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2H
"lstm_5/lstm_cell_10/ReadVariableOp"lstm_5/lstm_cell_10/ReadVariableOp2L
$lstm_5/lstm_cell_10/ReadVariableOp_1$lstm_5/lstm_cell_10/ReadVariableOp_12L
$lstm_5/lstm_cell_10/ReadVariableOp_2$lstm_5/lstm_cell_10/ReadVariableOp_22L
$lstm_5/lstm_cell_10/ReadVariableOp_3$lstm_5/lstm_cell_10/ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2T
(lstm_5/lstm_cell_10/split/ReadVariableOp(lstm_5/lstm_cell_10/split/ReadVariableOp2X
*lstm_5/lstm_cell_10/split_1/ReadVariableOp*lstm_5/lstm_cell_10/split_1/ReadVariableOp2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs
Э
ы
-__inference_sequential_5_layer_call_fn_189415

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1885512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs


'__inference_lstm_5_layer_call_fn_190123

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1880492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ&:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs
і	
н
D__inference_dense_17_layer_call_and_return_conditional_losses_188441

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
%

while_body_187561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_10_187585_0
while_lstm_cell_10_187587_0
while_lstm_cell_10_187589_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_10_187585
while_lstm_cell_10_187587
while_lstm_cell_10_187589Ђ*while/lstm_cell_10/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemс
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_187585_0while_lstm_cell_10_187587_0while_lstm_cell_10_187589_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_1870992,
*while/lstm_cell_10/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2К
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ф
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Ф
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_10_187585while_lstm_cell_10_187585_0"8
while_lstm_cell_10_187587while_lstm_cell_10_187587_0"8
while_lstm_cell_10_187589while_lstm_cell_10_187589_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ :џџџџџџџџџ : : :::2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
Эg
ў
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_191173

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
	ones_like^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_3g
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4g
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_5g
mul_6Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6g
mul_7Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_10л
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulе
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulд
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityи

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1з

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ&:џџџџџџџџџ :џџџџџџџџџ :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/1
2
Ю
H__inference_sequential_5_layer_call_and_return_conditional_losses_188470
lstm_5_input
lstm_5_188339
lstm_5_188341
lstm_5_188343
dense_15_188368
dense_15_188370
dense_16_188425
dense_16_188427
dense_17_188452
dense_17_188454
identityЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCallЂ!dropout_5/StatefulPartitionedCallЂlstm_5/StatefulPartitionedCallЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЁ
lstm_5/StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputlstm_5_188339lstm_5_188341lstm_5_188343*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1880492 
lstm_5/StatefulPartitionedCallЕ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_188368dense_15_188370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1883572"
 dense_15/StatefulPartitionedCall
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1883852#
!dropout_5/StatefulPartitionedCallИ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_16_188425dense_16_188427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1884142"
 dense_16/StatefulPartitionedCallЗ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_188452dense_17_188454*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1884412"
 dense_17/StatefulPartitionedCallЫ
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_188339*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulУ
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_188341*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulЇ
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:Y U
+
_output_shapes
:џџџџџџџџџ&
&
_user_specified_namelstm_5_input
Ћ
У
while_cond_187560
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_187560___redundant_placeholder04
0while_while_cond_187560___redundant_placeholder14
0while_while_cond_187560___redundant_placeholder24
0while_while_cond_187560___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
%

while_body_187417
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_10_187441_0
while_lstm_cell_10_187443_0
while_lstm_cell_10_187445_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_10_187441
while_lstm_cell_10_187443
while_lstm_cell_10_187445Ђ*while/lstm_cell_10/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemс
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_187441_0while_lstm_cell_10_187443_0while_lstm_cell_10_187445_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_1870032,
*while/lstm_cell_10/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2К
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ф
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4Ф
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_10_187441while_lstm_cell_10_187441_0"8
while_lstm_cell_10_187443while_lstm_cell_10_187443_0"8
while_lstm_cell_10_187445while_lstm_cell_10_187445_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ :џџџџџџџџџ : : :::2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
Л
Э
-__inference_lstm_cell_10_layer_call_fn_191207

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_1870992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ&:џџџџџџџџџ :џџџџџџџџџ :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/1
ў	
Я
lstm_5_while_cond_188861*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1B
>lstm_5_while_lstm_5_while_cond_188861___redundant_placeholder0B
>lstm_5_while_lstm_5_while_cond_188861___redundant_placeholder1B
>lstm_5_while_lstm_5_while_cond_188861___redundant_placeholder2B
>lstm_5_while_lstm_5_while_cond_188861___redundant_placeholder3
lstm_5_while_identity

lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
lstm_5/while/Lessr
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_5/while/Identity"7
lstm_5_while_identitylstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
оЕ
Й
B__inference_lstm_5_layer_call_and_return_conditional_losses_188316

inputs.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identityЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_10/ReadVariableOpЂlstm_cell_10/ReadVariableOp_1Ђlstm_cell_10/ReadVariableOp_2Ђlstm_cell_10/ReadVariableOp_3Ђ!lstm_cell_10/split/ReadVariableOpЂ#lstm_cell_10/split_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ&*
shrink_axis_mask2
strided_slice_2
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_10/ones_like/ConstИ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/ones_like~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
lstm_cell_10/ones_like_1/ConstР
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/ones_like_1
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul
lstm_cell_10/mul_1Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_1
lstm_cell_10/mul_2Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_2
lstm_cell_10/mul_3Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimВ
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02#
!lstm_cell_10/split/ReadVariableOpл
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_1
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_2
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimД
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_10/split_1/ReadVariableOpг
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1Ї
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd­
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_1­
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_2­
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_3
lstm_cell_10/mul_4Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_4
lstm_cell_10/mul_5Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_5
lstm_cell_10/mul_6Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_6
lstm_cell_10/mul_7Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_7 
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2Ъ
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_sliceЇ
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_4
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/SigmoidЄ
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_1
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2ж
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1Љ
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_5Ѕ
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_8Є
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_2
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2ж
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2Љ
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_6Ѕ
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_9
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_3Є
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_3
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2ж
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3Љ
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_7Ѕ
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_4
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh_1
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterф
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_188168*
condR
while_cond_188167*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeш
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulт
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulИ
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs
а0
Њ
H__inference_sequential_5_layer_call_and_return_conditional_losses_188509
lstm_5_input
lstm_5_188473
lstm_5_188475
lstm_5_188477
dense_15_188480
dense_15_188482
dense_16_188486
dense_16_188488
dense_17_188491
dense_17_188493
identityЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCallЂlstm_5/StatefulPartitionedCallЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЁ
lstm_5/StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputlstm_5_188473lstm_5_188475lstm_5_188477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1883162 
lstm_5/StatefulPartitionedCallЕ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_188480dense_15_188482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1883572"
 dense_15/StatefulPartitionedCallњ
dropout_5/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1883902
dropout_5/PartitionedCallА
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_16_188486dense_16_188488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1884142"
 dense_16/StatefulPartitionedCallЗ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_188491dense_17_188493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1884412"
 dense_17/StatefulPartitionedCallЫ
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_188473*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulУ
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_188475*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:Y U
+
_output_shapes
:џџџџџџџџџ&
&
_user_specified_namelstm_5_input
ШЏ
ў
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_191077

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeг
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2вЄ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeй
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ддЇ2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_1/GreaterEqual/yЦ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeй
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2гЈт2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_2/GreaterEqual/yЦ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeи
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ІK2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_3/GreaterEqual/yЦ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_3/Mul_1^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_4/Const
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shapeй
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ц2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_4/GreaterEqual/yЦ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_5/Const
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shapeй
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2б2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_5/GreaterEqual/yЦ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_6/Const
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shapeй
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Њ2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_6/GreaterEqual/yЦ
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_6/GreaterEqual
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_6/Cast
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_7/Const
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shapeи
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Є52(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_7/GreaterEqual/yЦ
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_7/GreaterEqual
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_7/Cast
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_3f
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4f
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_5f
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6f
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_10л
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulе
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulд
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityи

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1з

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ&:џџџџџџџџџ :џџџџџџџџџ :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/1
Ћ
У
while_cond_187836
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_187836___redundant_placeholder04
0while_while_cond_187836___redundant_placeholder14
0while_while_cond_187836___redundant_placeholder24
0while_while_cond_187836___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
Ћ
У
while_cond_187416
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_187416___redundant_placeholder04
0while_while_cond_187416___redundant_placeholder14
0while_while_cond_187416___redundant_placeholder24
0while_while_cond_187416___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
і	
н
D__inference_dense_17_layer_call_and_return_conditional_losses_190896

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Нg
ќ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_187099

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
	ones_like\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_3e
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4e
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_5e
mul_6Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6e
mul_7Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_10л
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulе
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulд
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityи

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1з

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ&:џџџџџџџџџ :џџџџџџџџџ :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates
№	
н
D__inference_dense_15_layer_call_and_return_conditional_losses_188357

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ћ
У
while_cond_190647
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_190647___redundant_placeholder04
0while_while_cond_190647___redundant_placeholder14
0while_while_cond_190647___redundant_placeholder24
0while_while_cond_190647___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
Ш
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_188390

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ћ
У
while_cond_188167
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_188167___redundant_placeholder04
0while_while_cond_188167___redundant_placeholder14
0while_while_cond_188167___redundant_placeholder24
0while_while_cond_188167___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
з[
з
B__inference_lstm_5_layer_call_and_return_conditional_losses_187498

inputs
lstm_cell_10_187404
lstm_cell_10_187406
lstm_cell_10_187408
identityЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂ$lstm_cell_10/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ&*
shrink_axis_mask2
strided_slice_2
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_187404lstm_cell_10_187406lstm_cell_10_187408*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_1870032&
$lstm_cell_10/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЃ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_187404lstm_cell_10_187406lstm_cell_10_187408*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_187417*
condR
while_cond_187416*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeб
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_10_187404*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulЩ
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_10_187406*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_10/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ&
 
_user_specified_nameinputs
Я
Л
B__inference_lstm_5_layer_call_and_return_conditional_losses_190529
inputs_0.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identityЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_10/ReadVariableOpЂlstm_cell_10/ReadVariableOp_1Ђlstm_cell_10/ReadVariableOp_2Ђlstm_cell_10/ReadVariableOp_3Ђ!lstm_cell_10/split/ReadVariableOpЂ#lstm_cell_10/split_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ&*
shrink_axis_mask2
strided_slice_2
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_10/ones_like/ConstИ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/ones_like}
lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout/ConstГ
lstm_cell_10/dropout/MulMullstm_cell_10/ones_like:output:0#lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout/Mul
lstm_cell_10/dropout/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout/Shapeљ
1lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2Ь23
1lstm_cell_10/dropout/random_uniform/RandomUniform
#lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2%
#lstm_cell_10/dropout/GreaterEqual/yђ
!lstm_cell_10/dropout/GreaterEqualGreaterEqual:lstm_cell_10/dropout/random_uniform/RandomUniform:output:0,lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2#
!lstm_cell_10/dropout/GreaterEqualІ
lstm_cell_10/dropout/CastCast%lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout/CastЎ
lstm_cell_10/dropout/Mul_1Mullstm_cell_10/dropout/Mul:z:0lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout/Mul_1
lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_1/ConstЙ
lstm_cell_10/dropout_1/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_1/Mul
lstm_cell_10/dropout_1/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_1/Shape
3lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2НЩ25
3lstm_cell_10/dropout_1/random_uniform/RandomUniform
%lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_1/GreaterEqual/yњ
#lstm_cell_10/dropout_1/GreaterEqualGreaterEqual<lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_cell_10/dropout_1/GreaterEqualЌ
lstm_cell_10/dropout_1/CastCast'lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_1/CastЖ
lstm_cell_10/dropout_1/Mul_1Mullstm_cell_10/dropout_1/Mul:z:0lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_1/Mul_1
lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_2/ConstЙ
lstm_cell_10/dropout_2/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_2/Mul
lstm_cell_10/dropout_2/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_2/Shape
3lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2Дд25
3lstm_cell_10/dropout_2/random_uniform/RandomUniform
%lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_2/GreaterEqual/yњ
#lstm_cell_10/dropout_2/GreaterEqualGreaterEqual<lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_cell_10/dropout_2/GreaterEqualЌ
lstm_cell_10/dropout_2/CastCast'lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_2/CastЖ
lstm_cell_10/dropout_2/Mul_1Mullstm_cell_10/dropout_2/Mul:z:0lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_2/Mul_1
lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_3/ConstЙ
lstm_cell_10/dropout_3/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_3/Mul
lstm_cell_10/dropout_3/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_3/Shapeџ
3lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2y25
3lstm_cell_10/dropout_3/random_uniform/RandomUniform
%lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_3/GreaterEqual/yњ
#lstm_cell_10/dropout_3/GreaterEqualGreaterEqual<lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_cell_10/dropout_3/GreaterEqualЌ
lstm_cell_10/dropout_3/CastCast'lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_3/CastЖ
lstm_cell_10/dropout_3/Mul_1Mullstm_cell_10/dropout_3/Mul:z:0lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_3/Mul_1~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
lstm_cell_10/ones_like_1/ConstР
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/ones_like_1
lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_4/ConstЛ
lstm_cell_10/dropout_4/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_4/Mul
lstm_cell_10/dropout_4/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_4/Shape
3lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2У25
3lstm_cell_10/dropout_4/random_uniform/RandomUniform
%lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_4/GreaterEqual/yњ
#lstm_cell_10/dropout_4/GreaterEqualGreaterEqual<lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_4/GreaterEqualЌ
lstm_cell_10/dropout_4/CastCast'lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_4/CastЖ
lstm_cell_10/dropout_4/Mul_1Mullstm_cell_10/dropout_4/Mul:z:0lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_4/Mul_1
lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_5/ConstЛ
lstm_cell_10/dropout_5/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_5/Mul
lstm_cell_10/dropout_5/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_5/Shape
3lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Џун25
3lstm_cell_10/dropout_5/random_uniform/RandomUniform
%lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_5/GreaterEqual/yњ
#lstm_cell_10/dropout_5/GreaterEqualGreaterEqual<lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_5/GreaterEqualЌ
lstm_cell_10/dropout_5/CastCast'lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_5/CastЖ
lstm_cell_10/dropout_5/Mul_1Mullstm_cell_10/dropout_5/Mul:z:0lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_5/Mul_1
lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_6/ConstЛ
lstm_cell_10/dropout_6/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_6/Mul
lstm_cell_10/dropout_6/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_6/Shapeџ
3lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2јёI25
3lstm_cell_10/dropout_6/random_uniform/RandomUniform
%lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_6/GreaterEqual/yњ
#lstm_cell_10/dropout_6/GreaterEqualGreaterEqual<lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_6/GreaterEqualЌ
lstm_cell_10/dropout_6/CastCast'lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_6/CastЖ
lstm_cell_10/dropout_6/Mul_1Mullstm_cell_10/dropout_6/Mul:z:0lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_6/Mul_1
lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_7/ConstЛ
lstm_cell_10/dropout_7/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_7/Mul
lstm_cell_10/dropout_7/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_7/Shapeџ
3lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2\25
3lstm_cell_10/dropout_7/random_uniform/RandomUniform
%lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_7/GreaterEqual/yњ
#lstm_cell_10/dropout_7/GreaterEqualGreaterEqual<lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_7/GreaterEqualЌ
lstm_cell_10/dropout_7/CastCast'lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_7/CastЖ
lstm_cell_10/dropout_7/Mul_1Mullstm_cell_10/dropout_7/Mul:z:0lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_7/Mul_1
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul
lstm_cell_10/mul_1Mulstrided_slice_2:output:0 lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_1
lstm_cell_10/mul_2Mulstrided_slice_2:output:0 lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_2
lstm_cell_10/mul_3Mulstrided_slice_2:output:0 lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimВ
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02#
!lstm_cell_10/split/ReadVariableOpл
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_1
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_2
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimД
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_10/split_1/ReadVariableOpг
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1Ї
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd­
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_1­
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_2­
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_3
lstm_cell_10/mul_4Mulzeros:output:0 lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_4
lstm_cell_10/mul_5Mulzeros:output:0 lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_5
lstm_cell_10/mul_6Mulzeros:output:0 lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_6
lstm_cell_10/mul_7Mulzeros:output:0 lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_7 
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2Ъ
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_sliceЇ
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_4
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/SigmoidЄ
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_1
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2ж
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1Љ
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_5Ѕ
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_8Є
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_2
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2ж
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2Љ
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_6Ѕ
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_9
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_3Є
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_3
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2ж
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3Љ
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_7Ѕ
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_4
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh_1
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterф
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_190317*
condR
while_cond_190316*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeш
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulт
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulИ
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ&
"
_user_specified_name
inputs/0
бЮ

%sequential_5_lstm_5_while_body_186645D
@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counterJ
Fsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations)
%sequential_5_lstm_5_while_placeholder+
'sequential_5_lstm_5_while_placeholder_1+
'sequential_5_lstm_5_while_placeholder_2+
'sequential_5_lstm_5_while_placeholder_3C
?sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1_0
{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0J
Fsequential_5_lstm_5_while_lstm_cell_10_split_readvariableop_resource_0L
Hsequential_5_lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0D
@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0&
"sequential_5_lstm_5_while_identity(
$sequential_5_lstm_5_while_identity_1(
$sequential_5_lstm_5_while_identity_2(
$sequential_5_lstm_5_while_identity_3(
$sequential_5_lstm_5_while_identity_4(
$sequential_5_lstm_5_while_identity_5A
=sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1}
ysequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensorH
Dsequential_5_lstm_5_while_lstm_cell_10_split_readvariableop_resourceJ
Fsequential_5_lstm_5_while_lstm_cell_10_split_1_readvariableop_resourceB
>sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resourceЂ5sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOpЂ7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_1Ђ7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_2Ђ7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3Ђ;sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOpЂ=sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOpы
Ksequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   2M
Ksequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeЫ
=sequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0%sequential_5_lstm_5_while_placeholderTsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ&*
element_dtype02?
=sequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItemф
6sequential_5/lstm_5/while/lstm_cell_10/ones_like/ShapeShapeDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:28
6sequential_5/lstm_5/while/lstm_cell_10/ones_like/ShapeЕ
6sequential_5/lstm_5/while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?28
6sequential_5/lstm_5/while/lstm_cell_10/ones_like/Const 
0sequential_5/lstm_5/while/lstm_cell_10/ones_likeFill?sequential_5/lstm_5/while/lstm_cell_10/ones_like/Shape:output:0?sequential_5/lstm_5/while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&22
0sequential_5/lstm_5/while/lstm_cell_10/ones_likeЫ
8sequential_5/lstm_5/while/lstm_cell_10/ones_like_1/ShapeShape'sequential_5_lstm_5_while_placeholder_2*
T0*
_output_shapes
:2:
8sequential_5/lstm_5/while/lstm_cell_10/ones_like_1/ShapeЙ
8sequential_5/lstm_5/while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8sequential_5/lstm_5/while/lstm_cell_10/ones_like_1/ConstЈ
2sequential_5/lstm_5/while/lstm_cell_10/ones_like_1FillAsequential_5/lstm_5/while/lstm_cell_10/ones_like_1/Shape:output:0Asequential_5/lstm_5/while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 24
2sequential_5/lstm_5/while/lstm_cell_10/ones_like_1
*sequential_5/lstm_5/while/lstm_cell_10/mulMulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_5/lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2,
*sequential_5/lstm_5/while/lstm_cell_10/mul
,sequential_5/lstm_5/while/lstm_cell_10/mul_1MulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_5/lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_1
,sequential_5/lstm_5/while/lstm_cell_10/mul_2MulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_5/lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_2
,sequential_5/lstm_5/while/lstm_cell_10/mul_3MulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_5/lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_3
,sequential_5/lstm_5/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_5/lstm_5/while/lstm_cell_10/ConstВ
6sequential_5/lstm_5/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential_5/lstm_5/while/lstm_cell_10/split/split_dim
;sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOpReadVariableOpFsequential_5_lstm_5_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&*
dtype02=
;sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOpУ
,sequential_5/lstm_5/while/lstm_cell_10/splitSplit?sequential_5/lstm_5/while/lstm_cell_10/split/split_dim:output:0Csequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2.
,sequential_5/lstm_5/while/lstm_cell_10/split
-sequential_5/lstm_5/while/lstm_cell_10/MatMulMatMul.sequential_5/lstm_5/while/lstm_cell_10/mul:z:05sequential_5/lstm_5/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_5/lstm_5/while/lstm_cell_10/MatMul
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_1MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_1:z:05sequential_5/lstm_5/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_1
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_2MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_2:z:05sequential_5/lstm_5/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_2
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_3MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_3:z:05sequential_5/lstm_5/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_3Ђ
.sequential_5/lstm_5/while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :20
.sequential_5/lstm_5/while/lstm_cell_10/Const_1Ж
8sequential_5/lstm_5/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8sequential_5/lstm_5/while/lstm_cell_10/split_1/split_dim
=sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOpHsequential_5_lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOpЛ
.sequential_5/lstm_5/while/lstm_cell_10/split_1SplitAsequential_5/lstm_5/while/lstm_cell_10/split_1/split_dim:output:0Esequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split20
.sequential_5/lstm_5/while/lstm_cell_10/split_1
.sequential_5/lstm_5/while/lstm_cell_10/BiasAddBiasAdd7sequential_5/lstm_5/while/lstm_cell_10/MatMul:product:07sequential_5/lstm_5/while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_5/lstm_5/while/lstm_cell_10/BiasAdd
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_1BiasAdd9sequential_5/lstm_5/while/lstm_cell_10/MatMul_1:product:07sequential_5/lstm_5/while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_1
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_2BiasAdd9sequential_5/lstm_5/while/lstm_cell_10/MatMul_2:product:07sequential_5/lstm_5/while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_2
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_3BiasAdd9sequential_5/lstm_5/while/lstm_cell_10/MatMul_3:product:07sequential_5/lstm_5/while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_3ћ
,sequential_5/lstm_5/while/lstm_cell_10/mul_4Mul'sequential_5_lstm_5_while_placeholder_2;sequential_5/lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_4ћ
,sequential_5/lstm_5/while/lstm_cell_10/mul_5Mul'sequential_5_lstm_5_while_placeholder_2;sequential_5/lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_5ћ
,sequential_5/lstm_5/while/lstm_cell_10/mul_6Mul'sequential_5_lstm_5_while_placeholder_2;sequential_5/lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_6ћ
,sequential_5/lstm_5/while/lstm_cell_10/mul_7Mul'sequential_5_lstm_5_while_placeholder_2;sequential_5/lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_7№
5sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOpReadVariableOp@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype027
5sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOpЩ
:sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stackЭ
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2>
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_1Э
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_2ц
4sequential_5/lstm_5/while/lstm_cell_10/strided_sliceStridedSlice=sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp:value:0Csequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack:output:0Esequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_1:output:0Esequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask26
4sequential_5/lstm_5/while/lstm_cell_10/strided_slice
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_4MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_4:z:0=sequential_5/lstm_5/while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_4
*sequential_5/lstm_5/while/lstm_cell_10/addAddV27sequential_5/lstm_5/while/lstm_cell_10/BiasAdd:output:09sequential_5/lstm_5/while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_5/lstm_5/while/lstm_cell_10/addЭ
.sequential_5/lstm_5/while/lstm_cell_10/SigmoidSigmoid.sequential_5/lstm_5/while/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 20
.sequential_5/lstm_5/while/lstm_cell_10/Sigmoidє
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_1ReadVariableOp@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype029
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_1Э
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2>
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stackб
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_1б
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_2ђ
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1StridedSlice?sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_1:value:0Esequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_1:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask28
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_5MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_5:z:0?sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_5
,sequential_5/lstm_5/while/lstm_cell_10/add_1AddV29sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_1:output:09sequential_5/lstm_5/while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_5/lstm_5/while/lstm_cell_10/add_1г
0sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_1Sigmoid0sequential_5/lstm_5/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_1є
,sequential_5/lstm_5/while/lstm_cell_10/mul_8Mul4sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_1:y:0'sequential_5_lstm_5_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_8є
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_2ReadVariableOp@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype029
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_2Э
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2>
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stackб
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_1б
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_2ђ
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2StridedSlice?sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_2:value:0Esequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_1:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask28
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_6MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_6:z:0?sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_6
,sequential_5/lstm_5/while/lstm_cell_10/add_2AddV29sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_2:output:09sequential_5/lstm_5/while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_5/lstm_5/while/lstm_cell_10/add_2Ц
+sequential_5/lstm_5/while/lstm_cell_10/TanhTanh0sequential_5/lstm_5/while/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2-
+sequential_5/lstm_5/while/lstm_cell_10/Tanhњ
,sequential_5/lstm_5/while/lstm_cell_10/mul_9Mul2sequential_5/lstm_5/while/lstm_cell_10/Sigmoid:y:0/sequential_5/lstm_5/while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_9ћ
,sequential_5/lstm_5/while/lstm_cell_10/add_3AddV20sequential_5/lstm_5/while/lstm_cell_10/mul_8:z:00sequential_5/lstm_5/while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_5/lstm_5/while/lstm_cell_10/add_3є
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3ReadVariableOp@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype029
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3Э
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2>
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stackб
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_1б
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_2ђ
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3StridedSlice?sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3:value:0Esequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_1:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask28
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_7MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_7:z:0?sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_7
,sequential_5/lstm_5/while/lstm_cell_10/add_4AddV29sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_3:output:09sequential_5/lstm_5/while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_5/lstm_5/while/lstm_cell_10/add_4г
0sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_2Sigmoid0sequential_5/lstm_5/while/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_2Ъ
-sequential_5/lstm_5/while/lstm_cell_10/Tanh_1Tanh0sequential_5/lstm_5/while/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_5/lstm_5/while/lstm_cell_10/Tanh_1
-sequential_5/lstm_5/while/lstm_cell_10/mul_10Mul4sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_2:y:01sequential_5/lstm_5/while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2/
-sequential_5/lstm_5/while/lstm_cell_10/mul_10Х
>sequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_5_lstm_5_while_placeholder_1%sequential_5_lstm_5_while_placeholder1sequential_5/lstm_5/while/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02@
>sequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem
sequential_5/lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_5/lstm_5/while/add/yЙ
sequential_5/lstm_5/while/addAddV2%sequential_5_lstm_5_while_placeholder(sequential_5/lstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_5/while/add
!sequential_5/lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_5/lstm_5/while/add_1/yк
sequential_5/lstm_5/while/add_1AddV2@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counter*sequential_5/lstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_5/while/add_1ў
"sequential_5/lstm_5/while/IdentityIdentity#sequential_5/lstm_5/while/add_1:z:06^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2$
"sequential_5/lstm_5/while/IdentityЅ
$sequential_5/lstm_5/while/Identity_1IdentityFsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations6^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_5/lstm_5/while/Identity_1
$sequential_5/lstm_5/while/Identity_2Identity!sequential_5/lstm_5/while/add:z:06^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_5/lstm_5/while/Identity_2­
$sequential_5/lstm_5/while/Identity_3IdentityNsequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:06^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_5/lstm_5/while/Identity_3Ё
$sequential_5/lstm_5/while/Identity_4Identity1sequential_5/lstm_5/while/lstm_cell_10/mul_10:z:06^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$sequential_5/lstm_5/while/Identity_4 
$sequential_5/lstm_5/while/Identity_5Identity0sequential_5/lstm_5/while/lstm_cell_10/add_3:z:06^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$sequential_5/lstm_5/while/Identity_5"Q
"sequential_5_lstm_5_while_identity+sequential_5/lstm_5/while/Identity:output:0"U
$sequential_5_lstm_5_while_identity_1-sequential_5/lstm_5/while/Identity_1:output:0"U
$sequential_5_lstm_5_while_identity_2-sequential_5/lstm_5/while/Identity_2:output:0"U
$sequential_5_lstm_5_while_identity_3-sequential_5/lstm_5/while/Identity_3:output:0"U
$sequential_5_lstm_5_while_identity_4-sequential_5/lstm_5/while/Identity_4:output:0"U
$sequential_5_lstm_5_while_identity_5-sequential_5/lstm_5/while/Identity_5:output:0"
>sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0"
Fsequential_5_lstm_5_while_lstm_cell_10_split_1_readvariableop_resourceHsequential_5_lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0"
Dsequential_5_lstm_5_while_lstm_cell_10_split_readvariableop_resourceFsequential_5_lstm_5_while_lstm_cell_10_split_readvariableop_resource_0"
=sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1?sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1_0"ј
ysequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ :џџџџџџџџџ : : :::2n
5sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp5sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp2r
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_17sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_12r
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_27sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_22r
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_37sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_32z
;sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp;sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp2~
=sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp=sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
п
ё
-__inference_sequential_5_layer_call_fn_188634
lstm_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1886132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:џџџџџџџџџ&
&
_user_specified_namelstm_5_input

Й
B__inference_lstm_5_layer_call_and_return_conditional_losses_188049

inputs.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identityЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_10/ReadVariableOpЂlstm_cell_10/ReadVariableOp_1Ђlstm_cell_10/ReadVariableOp_2Ђlstm_cell_10/ReadVariableOp_3Ђ!lstm_cell_10/split/ReadVariableOpЂ#lstm_cell_10/split_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ&*
shrink_axis_mask2
strided_slice_2
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_10/ones_like/ConstИ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/ones_like}
lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout/ConstГ
lstm_cell_10/dropout/MulMullstm_cell_10/ones_like:output:0#lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout/Mul
lstm_cell_10/dropout/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout/Shapeњ
1lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2Нк23
1lstm_cell_10/dropout/random_uniform/RandomUniform
#lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2%
#lstm_cell_10/dropout/GreaterEqual/yђ
!lstm_cell_10/dropout/GreaterEqualGreaterEqual:lstm_cell_10/dropout/random_uniform/RandomUniform:output:0,lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2#
!lstm_cell_10/dropout/GreaterEqualІ
lstm_cell_10/dropout/CastCast%lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout/CastЎ
lstm_cell_10/dropout/Mul_1Mullstm_cell_10/dropout/Mul:z:0lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout/Mul_1
lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_1/ConstЙ
lstm_cell_10/dropout_1/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_1/Mul
lstm_cell_10/dropout_1/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_1/Shape
3lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2хі25
3lstm_cell_10/dropout_1/random_uniform/RandomUniform
%lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_1/GreaterEqual/yњ
#lstm_cell_10/dropout_1/GreaterEqualGreaterEqual<lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_cell_10/dropout_1/GreaterEqualЌ
lstm_cell_10/dropout_1/CastCast'lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_1/CastЖ
lstm_cell_10/dropout_1/Mul_1Mullstm_cell_10/dropout_1/Mul:z:0lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_1/Mul_1
lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_2/ConstЙ
lstm_cell_10/dropout_2/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_2/Mul
lstm_cell_10/dropout_2/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_2/Shapeџ
3lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2Ы425
3lstm_cell_10/dropout_2/random_uniform/RandomUniform
%lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_2/GreaterEqual/yњ
#lstm_cell_10/dropout_2/GreaterEqualGreaterEqual<lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_cell_10/dropout_2/GreaterEqualЌ
lstm_cell_10/dropout_2/CastCast'lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_2/CastЖ
lstm_cell_10/dropout_2/Mul_1Mullstm_cell_10/dropout_2/Mul:z:0lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_2/Mul_1
lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_3/ConstЙ
lstm_cell_10/dropout_3/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_3/Mul
lstm_cell_10/dropout_3/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_3/Shape
3lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2їЭ25
3lstm_cell_10/dropout_3/random_uniform/RandomUniform
%lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_3/GreaterEqual/yњ
#lstm_cell_10/dropout_3/GreaterEqualGreaterEqual<lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_cell_10/dropout_3/GreaterEqualЌ
lstm_cell_10/dropout_3/CastCast'lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_3/CastЖ
lstm_cell_10/dropout_3/Mul_1Mullstm_cell_10/dropout_3/Mul:z:0lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_3/Mul_1~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
lstm_cell_10/ones_like_1/ConstР
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/ones_like_1
lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_4/ConstЛ
lstm_cell_10/dropout_4/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_4/Mul
lstm_cell_10/dropout_4/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_4/Shapeџ
3lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2щв025
3lstm_cell_10/dropout_4/random_uniform/RandomUniform
%lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_4/GreaterEqual/yњ
#lstm_cell_10/dropout_4/GreaterEqualGreaterEqual<lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_4/GreaterEqualЌ
lstm_cell_10/dropout_4/CastCast'lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_4/CastЖ
lstm_cell_10/dropout_4/Mul_1Mullstm_cell_10/dropout_4/Mul:z:0lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_4/Mul_1
lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_5/ConstЛ
lstm_cell_10/dropout_5/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_5/Mul
lstm_cell_10/dropout_5/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_5/Shapeџ
3lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2эX25
3lstm_cell_10/dropout_5/random_uniform/RandomUniform
%lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_5/GreaterEqual/yњ
#lstm_cell_10/dropout_5/GreaterEqualGreaterEqual<lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_5/GreaterEqualЌ
lstm_cell_10/dropout_5/CastCast'lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_5/CastЖ
lstm_cell_10/dropout_5/Mul_1Mullstm_cell_10/dropout_5/Mul:z:0lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_5/Mul_1
lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_6/ConstЛ
lstm_cell_10/dropout_6/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_6/Mul
lstm_cell_10/dropout_6/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_6/Shape
3lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Ъы25
3lstm_cell_10/dropout_6/random_uniform/RandomUniform
%lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_6/GreaterEqual/yњ
#lstm_cell_10/dropout_6/GreaterEqualGreaterEqual<lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_6/GreaterEqualЌ
lstm_cell_10/dropout_6/CastCast'lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_6/CastЖ
lstm_cell_10/dropout_6/Mul_1Mullstm_cell_10/dropout_6/Mul:z:0lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_6/Mul_1
lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_7/ConstЛ
lstm_cell_10/dropout_7/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_7/Mul
lstm_cell_10/dropout_7/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_7/Shape
3lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2чэИ25
3lstm_cell_10/dropout_7/random_uniform/RandomUniform
%lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_7/GreaterEqual/yњ
#lstm_cell_10/dropout_7/GreaterEqualGreaterEqual<lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_7/GreaterEqualЌ
lstm_cell_10/dropout_7/CastCast'lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_7/CastЖ
lstm_cell_10/dropout_7/Mul_1Mullstm_cell_10/dropout_7/Mul:z:0lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_7/Mul_1
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul
lstm_cell_10/mul_1Mulstrided_slice_2:output:0 lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_1
lstm_cell_10/mul_2Mulstrided_slice_2:output:0 lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_2
lstm_cell_10/mul_3Mulstrided_slice_2:output:0 lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimВ
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02#
!lstm_cell_10/split/ReadVariableOpл
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_1
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_2
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimД
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_10/split_1/ReadVariableOpг
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1Ї
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd­
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_1­
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_2­
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_3
lstm_cell_10/mul_4Mulzeros:output:0 lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_4
lstm_cell_10/mul_5Mulzeros:output:0 lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_5
lstm_cell_10/mul_6Mulzeros:output:0 lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_6
lstm_cell_10/mul_7Mulzeros:output:0 lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_7 
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2Ъ
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_sliceЇ
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_4
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/SigmoidЄ
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_1
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2ж
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1Љ
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_5Ѕ
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_8Є
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_2
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2ж
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2Љ
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_6Ѕ
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_9
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_3Є
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_3
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2ж
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3Љ
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_7Ѕ
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_4
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh_1
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterф
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_187837*
condR
while_cond_187836*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeш
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulт
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulИ
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs

Й
B__inference_lstm_5_layer_call_and_return_conditional_losses_189845

inputs.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identityЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_10/ReadVariableOpЂlstm_cell_10/ReadVariableOp_1Ђlstm_cell_10/ReadVariableOp_2Ђlstm_cell_10/ReadVariableOp_3Ђ!lstm_cell_10/split/ReadVariableOpЂ#lstm_cell_10/split_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ&*
shrink_axis_mask2
strided_slice_2
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_10/ones_like/ConstИ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/ones_like}
lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout/ConstГ
lstm_cell_10/dropout/MulMullstm_cell_10/ones_like:output:0#lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout/Mul
lstm_cell_10/dropout/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout/Shapeњ
1lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2И§џ23
1lstm_cell_10/dropout/random_uniform/RandomUniform
#lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2%
#lstm_cell_10/dropout/GreaterEqual/yђ
!lstm_cell_10/dropout/GreaterEqualGreaterEqual:lstm_cell_10/dropout/random_uniform/RandomUniform:output:0,lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2#
!lstm_cell_10/dropout/GreaterEqualІ
lstm_cell_10/dropout/CastCast%lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout/CastЎ
lstm_cell_10/dropout/Mul_1Mullstm_cell_10/dropout/Mul:z:0lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout/Mul_1
lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_1/ConstЙ
lstm_cell_10/dropout_1/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_1/Mul
lstm_cell_10/dropout_1/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_1/Shape
3lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2гњ25
3lstm_cell_10/dropout_1/random_uniform/RandomUniform
%lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_1/GreaterEqual/yњ
#lstm_cell_10/dropout_1/GreaterEqualGreaterEqual<lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_cell_10/dropout_1/GreaterEqualЌ
lstm_cell_10/dropout_1/CastCast'lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_1/CastЖ
lstm_cell_10/dropout_1/Mul_1Mullstm_cell_10/dropout_1/Mul:z:0lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_1/Mul_1
lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_2/ConstЙ
lstm_cell_10/dropout_2/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_2/Mul
lstm_cell_10/dropout_2/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_2/Shape
3lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2Ѓо25
3lstm_cell_10/dropout_2/random_uniform/RandomUniform
%lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_2/GreaterEqual/yњ
#lstm_cell_10/dropout_2/GreaterEqualGreaterEqual<lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_cell_10/dropout_2/GreaterEqualЌ
lstm_cell_10/dropout_2/CastCast'lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_2/CastЖ
lstm_cell_10/dropout_2/Mul_1Mullstm_cell_10/dropout_2/Mul:z:0lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_2/Mul_1
lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_3/ConstЙ
lstm_cell_10/dropout_3/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_3/Mul
lstm_cell_10/dropout_3/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_3/Shape
3lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2Й25
3lstm_cell_10/dropout_3/random_uniform/RandomUniform
%lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_3/GreaterEqual/yњ
#lstm_cell_10/dropout_3/GreaterEqualGreaterEqual<lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_cell_10/dropout_3/GreaterEqualЌ
lstm_cell_10/dropout_3/CastCast'lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_3/CastЖ
lstm_cell_10/dropout_3/Mul_1Mullstm_cell_10/dropout_3/Mul:z:0lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/dropout_3/Mul_1~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
lstm_cell_10/ones_like_1/ConstР
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/ones_like_1
lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_4/ConstЛ
lstm_cell_10/dropout_4/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_4/Mul
lstm_cell_10/dropout_4/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_4/Shapeџ
3lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ФВ25
3lstm_cell_10/dropout_4/random_uniform/RandomUniform
%lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_4/GreaterEqual/yњ
#lstm_cell_10/dropout_4/GreaterEqualGreaterEqual<lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_4/GreaterEqualЌ
lstm_cell_10/dropout_4/CastCast'lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_4/CastЖ
lstm_cell_10/dropout_4/Mul_1Mullstm_cell_10/dropout_4/Mul:z:0lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_4/Mul_1
lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_5/ConstЛ
lstm_cell_10/dropout_5/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_5/Mul
lstm_cell_10/dropout_5/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_5/Shape
3lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2хі25
3lstm_cell_10/dropout_5/random_uniform/RandomUniform
%lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_5/GreaterEqual/yњ
#lstm_cell_10/dropout_5/GreaterEqualGreaterEqual<lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_5/GreaterEqualЌ
lstm_cell_10/dropout_5/CastCast'lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_5/CastЖ
lstm_cell_10/dropout_5/Mul_1Mullstm_cell_10/dropout_5/Mul:z:0lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_5/Mul_1
lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_6/ConstЛ
lstm_cell_10/dropout_6/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_6/Mul
lstm_cell_10/dropout_6/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_6/Shape
3lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Ќд25
3lstm_cell_10/dropout_6/random_uniform/RandomUniform
%lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_6/GreaterEqual/yњ
#lstm_cell_10/dropout_6/GreaterEqualGreaterEqual<lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_6/GreaterEqualЌ
lstm_cell_10/dropout_6/CastCast'lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_6/CastЖ
lstm_cell_10/dropout_6/Mul_1Mullstm_cell_10/dropout_6/Mul:z:0lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_6/Mul_1
lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell_10/dropout_7/ConstЛ
lstm_cell_10/dropout_7/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_7/Mul
lstm_cell_10/dropout_7/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_7/Shape
3lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2§Ш25
3lstm_cell_10/dropout_7/random_uniform/RandomUniform
%lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2'
%lstm_cell_10/dropout_7/GreaterEqual/yњ
#lstm_cell_10/dropout_7/GreaterEqualGreaterEqual<lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_cell_10/dropout_7/GreaterEqualЌ
lstm_cell_10/dropout_7/CastCast'lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_7/CastЖ
lstm_cell_10/dropout_7/Mul_1Mullstm_cell_10/dropout_7/Mul:z:0lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/dropout_7/Mul_1
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul
lstm_cell_10/mul_1Mulstrided_slice_2:output:0 lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_1
lstm_cell_10/mul_2Mulstrided_slice_2:output:0 lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_2
lstm_cell_10/mul_3Mulstrided_slice_2:output:0 lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimВ
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02#
!lstm_cell_10/split/ReadVariableOpл
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_1
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_2
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimД
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_10/split_1/ReadVariableOpг
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1Ї
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd­
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_1­
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_2­
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_3
lstm_cell_10/mul_4Mulzeros:output:0 lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_4
lstm_cell_10/mul_5Mulzeros:output:0 lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_5
lstm_cell_10/mul_6Mulzeros:output:0 lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_6
lstm_cell_10/mul_7Mulzeros:output:0 lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_7 
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2Ъ
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_sliceЇ
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_4
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/SigmoidЄ
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_1
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2ж
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1Љ
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_5Ѕ
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_8Є
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_2
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2ж
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2Љ
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_6Ѕ
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_9
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_3Є
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_3
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2ж
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3Љ
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_7Ѕ
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_4
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh_1
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterф
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_189633*
condR
while_cond_189632*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeш
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulт
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulИ
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs
Ж
Л
B__inference_lstm_5_layer_call_and_return_conditional_losses_190796
inputs_0.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identityЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_10/ReadVariableOpЂlstm_cell_10/ReadVariableOp_1Ђlstm_cell_10/ReadVariableOp_2Ђlstm_cell_10/ReadVariableOp_3Ђ!lstm_cell_10/split/ReadVariableOpЂ#lstm_cell_10/split_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ&*
shrink_axis_mask2
strided_slice_2
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_10/ones_like/ConstИ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/ones_like~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
lstm_cell_10/ones_like_1/ConstР
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/ones_like_1
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul
lstm_cell_10/mul_1Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_1
lstm_cell_10/mul_2Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_2
lstm_cell_10/mul_3Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimВ
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02#
!lstm_cell_10/split/ReadVariableOpл
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_1
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_2
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimД
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_10/split_1/ReadVariableOpг
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1Ї
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd­
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_1­
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_2­
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_3
lstm_cell_10/mul_4Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_4
lstm_cell_10/mul_5Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_5
lstm_cell_10/mul_6Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_6
lstm_cell_10/mul_7Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_7 
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2Ъ
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_sliceЇ
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_4
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/SigmoidЄ
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_1
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2ж
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1Љ
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_5Ѕ
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_8Є
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_2
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2ж
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2Љ
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_6Ѕ
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_9
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_3Є
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_3
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2ж
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3Љ
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_7Ѕ
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_4
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh_1
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterф
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_190648*
condR
while_cond_190647*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeш
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulт
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulИ
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ&
"
_user_specified_name
inputs/0
м
~
)__inference_dense_17_layer_call_fn_190905

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1884412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


'__inference_lstm_5_layer_call_fn_190134

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1883162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ&:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs
Ћ
У
while_cond_189632
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_189632___redundant_placeholder04
0while_while_cond_189632___redundant_placeholder14
0while_while_cond_189632___redundant_placeholder24
0while_while_cond_189632___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
м
~
)__inference_dense_15_layer_call_fn_190838

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1883572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
оЕ
Й
B__inference_lstm_5_layer_call_and_return_conditional_losses_190112

inputs.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identityЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂlstm_cell_10/ReadVariableOpЂlstm_cell_10/ReadVariableOp_1Ђlstm_cell_10/ReadVariableOp_2Ђlstm_cell_10/ReadVariableOp_3Ђ!lstm_cell_10/split/ReadVariableOpЂ#lstm_cell_10/split_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ&*
shrink_axis_mask2
strided_slice_2
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_10/ones_like/ConstИ
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/ones_like~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
lstm_cell_10/ones_like_1/ConstР
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/ones_like_1
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul
lstm_cell_10/mul_1Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_1
lstm_cell_10/mul_2Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_2
lstm_cell_10/mul_3Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimВ
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02#
!lstm_cell_10/split/ReadVariableOpл
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_1
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_2
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dimД
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_10/split_1/ReadVariableOpг
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1Ї
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd­
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_1­
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_2­
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/BiasAdd_3
lstm_cell_10/mul_4Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_4
lstm_cell_10/mul_5Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_5
lstm_cell_10/mul_6Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_6
lstm_cell_10/mul_7Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_7 
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2Ъ
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_sliceЇ
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_4
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/SigmoidЄ
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_1
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2ж
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1Љ
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_5Ѕ
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_8Є
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_2
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2ж
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2Љ
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_6Ѕ
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_9
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_3Є
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02
lstm_cell_10/ReadVariableOp_3
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2ж
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3Љ
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/MatMul_7Ѕ
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/add_4
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/Tanh_1
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_cell_10/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterф
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_189964*
condR
while_cond_189963*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeш
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulт
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulИ
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs
ќй
п
H__inference_sequential_5_layer_call_and_return_conditional_losses_189103

inputs5
1lstm_5_lstm_cell_10_split_readvariableop_resource7
3lstm_5_lstm_cell_10_split_1_readvariableop_resource/
+lstm_5_lstm_cell_10_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identityЂdense_15/BiasAdd/ReadVariableOpЂdense_15/MatMul/ReadVariableOpЂdense_16/BiasAdd/ReadVariableOpЂdense_16/MatMul/ReadVariableOpЂdense_17/BiasAdd/ReadVariableOpЂdense_17/MatMul/ReadVariableOpЂ"lstm_5/lstm_cell_10/ReadVariableOpЂ$lstm_5/lstm_cell_10/ReadVariableOp_1Ђ$lstm_5/lstm_cell_10/ReadVariableOp_2Ђ$lstm_5/lstm_cell_10/ReadVariableOp_3Ђ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂ(lstm_5/lstm_cell_10/split/ReadVariableOpЂ*lstm_5/lstm_cell_10/split_1/ReadVariableOpЂlstm_5/whileR
lstm_5/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_5/Shape
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros/mul/y
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_5/zeros/Less/y
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros/packed/1
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros_1/mul/y
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_5/zeros_1/Less/y
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros_1/packed/1Ѕ
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/zeros_1
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose/perm
lstm_5/transpose	Transposeinputslstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ&2
lstm_5/transposed
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:2
lstm_5/Shape_1
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_1/stack
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_1
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_2
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slice_1
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"lstm_5/TensorArrayV2/element_shapeЮ
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2Э
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   2>
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_5/TensorArrayUnstack/TensorListFromTensor
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_2/stack
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_1
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_2І
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ&*
shrink_axis_mask2
lstm_5/strided_slice_2
#lstm_5/lstm_cell_10/ones_like/ShapeShapelstm_5/strided_slice_2:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/ones_like/Shape
#lstm_5/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_5/lstm_cell_10/ones_like/Constд
lstm_5/lstm_cell_10/ones_likeFill,lstm_5/lstm_cell_10/ones_like/Shape:output:0,lstm_5/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/lstm_cell_10/ones_like
!lstm_5/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!lstm_5/lstm_cell_10/dropout/ConstЯ
lstm_5/lstm_cell_10/dropout/MulMul&lstm_5/lstm_cell_10/ones_like:output:0*lstm_5/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2!
lstm_5/lstm_cell_10/dropout/Mul
!lstm_5/lstm_cell_10/dropout/ShapeShape&lstm_5/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2#
!lstm_5/lstm_cell_10/dropout/Shape
8lstm_5/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform*lstm_5/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2г2:
8lstm_5/lstm_cell_10/dropout/random_uniform/RandomUniform
*lstm_5/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2,
*lstm_5/lstm_cell_10/dropout/GreaterEqual/y
(lstm_5/lstm_cell_10/dropout/GreaterEqualGreaterEqualAlstm_5/lstm_cell_10/dropout/random_uniform/RandomUniform:output:03lstm_5/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2*
(lstm_5/lstm_cell_10/dropout/GreaterEqualЛ
 lstm_5/lstm_cell_10/dropout/CastCast,lstm_5/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2"
 lstm_5/lstm_cell_10/dropout/CastЪ
!lstm_5/lstm_cell_10/dropout/Mul_1Mul#lstm_5/lstm_cell_10/dropout/Mul:z:0$lstm_5/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2#
!lstm_5/lstm_cell_10/dropout/Mul_1
#lstm_5/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#lstm_5/lstm_cell_10/dropout_1/Constе
!lstm_5/lstm_cell_10/dropout_1/MulMul&lstm_5/lstm_cell_10/ones_like:output:0,lstm_5/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2#
!lstm_5/lstm_cell_10/dropout_1/Mul 
#lstm_5/lstm_cell_10/dropout_1/ShapeShape&lstm_5/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_1/Shape
:lstm_5/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2єЊ2<
:lstm_5/lstm_cell_10/dropout_1/random_uniform/RandomUniformЁ
,lstm_5/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2.
,lstm_5/lstm_cell_10/dropout_1/GreaterEqual/y
*lstm_5/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2,
*lstm_5/lstm_cell_10/dropout_1/GreaterEqualС
"lstm_5/lstm_cell_10/dropout_1/CastCast.lstm_5/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2$
"lstm_5/lstm_cell_10/dropout_1/Castв
#lstm_5/lstm_cell_10/dropout_1/Mul_1Mul%lstm_5/lstm_cell_10/dropout_1/Mul:z:0&lstm_5/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_5/lstm_cell_10/dropout_1/Mul_1
#lstm_5/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#lstm_5/lstm_cell_10/dropout_2/Constе
!lstm_5/lstm_cell_10/dropout_2/MulMul&lstm_5/lstm_cell_10/ones_like:output:0,lstm_5/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2#
!lstm_5/lstm_cell_10/dropout_2/Mul 
#lstm_5/lstm_cell_10/dropout_2/ShapeShape&lstm_5/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_2/Shape
:lstm_5/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2х2<
:lstm_5/lstm_cell_10/dropout_2/random_uniform/RandomUniformЁ
,lstm_5/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2.
,lstm_5/lstm_cell_10/dropout_2/GreaterEqual/y
*lstm_5/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2,
*lstm_5/lstm_cell_10/dropout_2/GreaterEqualС
"lstm_5/lstm_cell_10/dropout_2/CastCast.lstm_5/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2$
"lstm_5/lstm_cell_10/dropout_2/Castв
#lstm_5/lstm_cell_10/dropout_2/Mul_1Mul%lstm_5/lstm_cell_10/dropout_2/Mul:z:0&lstm_5/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_5/lstm_cell_10/dropout_2/Mul_1
#lstm_5/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#lstm_5/lstm_cell_10/dropout_3/Constе
!lstm_5/lstm_cell_10/dropout_3/MulMul&lstm_5/lstm_cell_10/ones_like:output:0,lstm_5/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2#
!lstm_5/lstm_cell_10/dropout_3/Mul 
#lstm_5/lstm_cell_10/dropout_3/ShapeShape&lstm_5/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_3/Shape
:lstm_5/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ѓЛЪ2<
:lstm_5/lstm_cell_10/dropout_3/random_uniform/RandomUniformЁ
,lstm_5/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2.
,lstm_5/lstm_cell_10/dropout_3/GreaterEqual/y
*lstm_5/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2,
*lstm_5/lstm_cell_10/dropout_3/GreaterEqualС
"lstm_5/lstm_cell_10/dropout_3/CastCast.lstm_5/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2$
"lstm_5/lstm_cell_10/dropout_3/Castв
#lstm_5/lstm_cell_10/dropout_3/Mul_1Mul%lstm_5/lstm_cell_10/dropout_3/Mul:z:0&lstm_5/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_5/lstm_cell_10/dropout_3/Mul_1
%lstm_5/lstm_cell_10/ones_like_1/ShapeShapelstm_5/zeros:output:0*
T0*
_output_shapes
:2'
%lstm_5/lstm_cell_10/ones_like_1/Shape
%lstm_5/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%lstm_5/lstm_cell_10/ones_like_1/Constм
lstm_5/lstm_cell_10/ones_like_1Fill.lstm_5/lstm_cell_10/ones_like_1/Shape:output:0.lstm_5/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/lstm_cell_10/ones_like_1
#lstm_5/lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#lstm_5/lstm_cell_10/dropout_4/Constз
!lstm_5/lstm_cell_10/dropout_4/MulMul(lstm_5/lstm_cell_10/ones_like_1:output:0,lstm_5/lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_5/lstm_cell_10/dropout_4/MulЂ
#lstm_5/lstm_cell_10/dropout_4/ShapeShape(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_4/Shape
:lstm_5/lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2зМЖ2<
:lstm_5/lstm_cell_10/dropout_4/random_uniform/RandomUniformЁ
,lstm_5/lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2.
,lstm_5/lstm_cell_10/dropout_4/GreaterEqual/y
*lstm_5/lstm_cell_10/dropout_4/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_5/lstm_cell_10/dropout_4/GreaterEqualС
"lstm_5/lstm_cell_10/dropout_4/CastCast.lstm_5/lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/lstm_cell_10/dropout_4/Castв
#lstm_5/lstm_cell_10/dropout_4/Mul_1Mul%lstm_5/lstm_cell_10/dropout_4/Mul:z:0&lstm_5/lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/lstm_cell_10/dropout_4/Mul_1
#lstm_5/lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#lstm_5/lstm_cell_10/dropout_5/Constз
!lstm_5/lstm_cell_10/dropout_5/MulMul(lstm_5/lstm_cell_10/ones_like_1:output:0,lstm_5/lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_5/lstm_cell_10/dropout_5/MulЂ
#lstm_5/lstm_cell_10/dropout_5/ShapeShape(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_5/Shape
:lstm_5/lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2еТМ2<
:lstm_5/lstm_cell_10/dropout_5/random_uniform/RandomUniformЁ
,lstm_5/lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2.
,lstm_5/lstm_cell_10/dropout_5/GreaterEqual/y
*lstm_5/lstm_cell_10/dropout_5/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_5/lstm_cell_10/dropout_5/GreaterEqualС
"lstm_5/lstm_cell_10/dropout_5/CastCast.lstm_5/lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/lstm_cell_10/dropout_5/Castв
#lstm_5/lstm_cell_10/dropout_5/Mul_1Mul%lstm_5/lstm_cell_10/dropout_5/Mul:z:0&lstm_5/lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/lstm_cell_10/dropout_5/Mul_1
#lstm_5/lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#lstm_5/lstm_cell_10/dropout_6/Constз
!lstm_5/lstm_cell_10/dropout_6/MulMul(lstm_5/lstm_cell_10/ones_like_1:output:0,lstm_5/lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_5/lstm_cell_10/dropout_6/MulЂ
#lstm_5/lstm_cell_10/dropout_6/ShapeShape(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_6/Shape
:lstm_5/lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Єнў2<
:lstm_5/lstm_cell_10/dropout_6/random_uniform/RandomUniformЁ
,lstm_5/lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2.
,lstm_5/lstm_cell_10/dropout_6/GreaterEqual/y
*lstm_5/lstm_cell_10/dropout_6/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_5/lstm_cell_10/dropout_6/GreaterEqualС
"lstm_5/lstm_cell_10/dropout_6/CastCast.lstm_5/lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/lstm_cell_10/dropout_6/Castв
#lstm_5/lstm_cell_10/dropout_6/Mul_1Mul%lstm_5/lstm_cell_10/dropout_6/Mul:z:0&lstm_5/lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/lstm_cell_10/dropout_6/Mul_1
#lstm_5/lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#lstm_5/lstm_cell_10/dropout_7/Constз
!lstm_5/lstm_cell_10/dropout_7/MulMul(lstm_5/lstm_cell_10/ones_like_1:output:0,lstm_5/lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_5/lstm_cell_10/dropout_7/MulЂ
#lstm_5/lstm_cell_10/dropout_7/ShapeShape(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_7/Shape
:lstm_5/lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2§§С2<
:lstm_5/lstm_cell_10/dropout_7/random_uniform/RandomUniformЁ
,lstm_5/lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2.
,lstm_5/lstm_cell_10/dropout_7/GreaterEqual/y
*lstm_5/lstm_cell_10/dropout_7/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*lstm_5/lstm_cell_10/dropout_7/GreaterEqualС
"lstm_5/lstm_cell_10/dropout_7/CastCast.lstm_5/lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/lstm_cell_10/dropout_7/Castв
#lstm_5/lstm_cell_10/dropout_7/Mul_1Mul%lstm_5/lstm_cell_10/dropout_7/Mul:z:0&lstm_5/lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/lstm_cell_10/dropout_7/Mul_1Г
lstm_5/lstm_cell_10/mulMullstm_5/strided_slice_2:output:0%lstm_5/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/lstm_cell_10/mulЙ
lstm_5/lstm_cell_10/mul_1Mullstm_5/strided_slice_2:output:0'lstm_5/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/lstm_cell_10/mul_1Й
lstm_5/lstm_cell_10/mul_2Mullstm_5/strided_slice_2:output:0'lstm_5/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/lstm_cell_10/mul_2Й
lstm_5/lstm_cell_10/mul_3Mullstm_5/strided_slice_2:output:0'lstm_5/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/lstm_cell_10/mul_3x
lstm_5/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_10/Const
#lstm_5/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_5/lstm_cell_10/split/split_dimЧ
(lstm_5/lstm_cell_10/split/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02*
(lstm_5/lstm_cell_10/split/ReadVariableOpї
lstm_5/lstm_cell_10/splitSplit,lstm_5/lstm_cell_10/split/split_dim:output:00lstm_5/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_5/lstm_cell_10/splitЕ
lstm_5/lstm_cell_10/MatMulMatMullstm_5/lstm_cell_10/mul:z:0"lstm_5/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMulЛ
lstm_5/lstm_cell_10/MatMul_1MatMullstm_5/lstm_cell_10/mul_1:z:0"lstm_5/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_1Л
lstm_5/lstm_cell_10/MatMul_2MatMullstm_5/lstm_cell_10/mul_2:z:0"lstm_5/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_2Л
lstm_5/lstm_cell_10/MatMul_3MatMullstm_5/lstm_cell_10/mul_3:z:0"lstm_5/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_3|
lstm_5/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_10/Const_1
%lstm_5/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%lstm_5/lstm_cell_10/split_1/split_dimЩ
*lstm_5/lstm_cell_10/split_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02,
*lstm_5/lstm_cell_10/split_1/ReadVariableOpя
lstm_5/lstm_cell_10/split_1Split.lstm_5/lstm_cell_10/split_1/split_dim:output:02lstm_5/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_5/lstm_cell_10/split_1У
lstm_5/lstm_cell_10/BiasAddBiasAdd$lstm_5/lstm_cell_10/MatMul:product:0$lstm_5/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/BiasAddЩ
lstm_5/lstm_cell_10/BiasAdd_1BiasAdd&lstm_5/lstm_cell_10/MatMul_1:product:0$lstm_5/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/BiasAdd_1Щ
lstm_5/lstm_cell_10/BiasAdd_2BiasAdd&lstm_5/lstm_cell_10/MatMul_2:product:0$lstm_5/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/BiasAdd_2Щ
lstm_5/lstm_cell_10/BiasAdd_3BiasAdd&lstm_5/lstm_cell_10/MatMul_3:product:0$lstm_5/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/BiasAdd_3Џ
lstm_5/lstm_cell_10/mul_4Mullstm_5/zeros:output:0'lstm_5/lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_4Џ
lstm_5/lstm_cell_10/mul_5Mullstm_5/zeros:output:0'lstm_5/lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_5Џ
lstm_5/lstm_cell_10/mul_6Mullstm_5/zeros:output:0'lstm_5/lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_6Џ
lstm_5/lstm_cell_10/mul_7Mullstm_5/zeros:output:0'lstm_5/lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_7Е
"lstm_5/lstm_cell_10/ReadVariableOpReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"lstm_5/lstm_cell_10/ReadVariableOpЃ
'lstm_5/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'lstm_5/lstm_cell_10/strided_slice/stackЇ
)lstm_5/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_5/lstm_cell_10/strided_slice/stack_1Ї
)lstm_5/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)lstm_5/lstm_cell_10/strided_slice/stack_2є
!lstm_5/lstm_cell_10/strided_sliceStridedSlice*lstm_5/lstm_cell_10/ReadVariableOp:value:00lstm_5/lstm_cell_10/strided_slice/stack:output:02lstm_5/lstm_cell_10/strided_slice/stack_1:output:02lstm_5/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!lstm_5/lstm_cell_10/strided_sliceУ
lstm_5/lstm_cell_10/MatMul_4MatMullstm_5/lstm_cell_10/mul_4:z:0*lstm_5/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_4Л
lstm_5/lstm_cell_10/addAddV2$lstm_5/lstm_cell_10/BiasAdd:output:0&lstm_5/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/add
lstm_5/lstm_cell_10/SigmoidSigmoidlstm_5/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/SigmoidЙ
$lstm_5/lstm_cell_10/ReadVariableOp_1ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_1Ї
)lstm_5/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_5/lstm_cell_10/strided_slice_1/stackЋ
+lstm_5/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+lstm_5/lstm_cell_10/strided_slice_1/stack_1Ћ
+lstm_5/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_1/stack_2
#lstm_5/lstm_cell_10/strided_slice_1StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_1:value:02lstm_5/lstm_cell_10/strided_slice_1/stack:output:04lstm_5/lstm_cell_10/strided_slice_1/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_1Х
lstm_5/lstm_cell_10/MatMul_5MatMullstm_5/lstm_cell_10/mul_5:z:0,lstm_5/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_5С
lstm_5/lstm_cell_10/add_1AddV2&lstm_5/lstm_cell_10/BiasAdd_1:output:0&lstm_5/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/add_1
lstm_5/lstm_cell_10/Sigmoid_1Sigmoidlstm_5/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/Sigmoid_1Ћ
lstm_5/lstm_cell_10/mul_8Mul!lstm_5/lstm_cell_10/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_8Й
$lstm_5/lstm_cell_10/ReadVariableOp_2ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_2Ї
)lstm_5/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)lstm_5/lstm_cell_10/strided_slice_2/stackЋ
+lstm_5/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+lstm_5/lstm_cell_10/strided_slice_2/stack_1Ћ
+lstm_5/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_2/stack_2
#lstm_5/lstm_cell_10/strided_slice_2StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_2:value:02lstm_5/lstm_cell_10/strided_slice_2/stack:output:04lstm_5/lstm_cell_10/strided_slice_2/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_2Х
lstm_5/lstm_cell_10/MatMul_6MatMullstm_5/lstm_cell_10/mul_6:z:0,lstm_5/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_6С
lstm_5/lstm_cell_10/add_2AddV2&lstm_5/lstm_cell_10/BiasAdd_2:output:0&lstm_5/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/add_2
lstm_5/lstm_cell_10/TanhTanhlstm_5/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/TanhЎ
lstm_5/lstm_cell_10/mul_9Mullstm_5/lstm_cell_10/Sigmoid:y:0lstm_5/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_9Џ
lstm_5/lstm_cell_10/add_3AddV2lstm_5/lstm_cell_10/mul_8:z:0lstm_5/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/add_3Й
$lstm_5/lstm_cell_10/ReadVariableOp_3ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_3Ї
)lstm_5/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)lstm_5/lstm_cell_10/strided_slice_3/stackЋ
+lstm_5/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_5/lstm_cell_10/strided_slice_3/stack_1Ћ
+lstm_5/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_3/stack_2
#lstm_5/lstm_cell_10/strided_slice_3StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_3:value:02lstm_5/lstm_cell_10/strided_slice_3/stack:output:04lstm_5/lstm_cell_10/strided_slice_3/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_3Х
lstm_5/lstm_cell_10/MatMul_7MatMullstm_5/lstm_cell_10/mul_7:z:0,lstm_5/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/MatMul_7С
lstm_5/lstm_cell_10/add_4AddV2&lstm_5/lstm_cell_10/BiasAdd_3:output:0&lstm_5/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/add_4
lstm_5/lstm_cell_10/Sigmoid_2Sigmoidlstm_5/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/Sigmoid_2
lstm_5/lstm_cell_10/Tanh_1Tanhlstm_5/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/Tanh_1Д
lstm_5/lstm_cell_10/mul_10Mul!lstm_5/lstm_cell_10/Sigmoid_2:y:0lstm_5/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/lstm_cell_10/mul_10
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2&
$lstm_5/TensorArrayV2_1/element_shapeд
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2_1\
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/time
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
lstm_5/while/maximum_iterationsx
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/while/loop_counterЭ
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_10_split_readvariableop_resource3lstm_5_lstm_cell_10_split_1_readvariableop_resource+lstm_5_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_5_while_body_188862*$
condR
lstm_5_while_cond_188861*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
lstm_5/whileУ
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    29
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02+
)lstm_5/TensorArrayV2Stack/TensorListStack
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_5/strided_slice_3/stack
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_5/strided_slice_3/stack_1
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_3/stack_2Ф
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
lstm_5/strided_slice_3
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose_1/permС
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
lstm_5/transpose_1t
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/runtimeЈ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_15/MatMul/ReadVariableOpЇ
dense_15/MatMulMatMullstm_5/strided_slice_3:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_15/MatMulЇ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_15/BiasAdd/ReadVariableOpЅ
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_15/BiasAdd|
dense_15/SigmoidSigmoiddense_15/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_15/Sigmoidw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
dropout_5/dropout/Const
dropout_5/dropout/MulMuldense_15/Sigmoid:y:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/dropout/Mulv
dropout_5/dropout/ShapeShapedense_15/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shapeв
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype020
.dropout_5/dropout/random_uniform/RandomUniform
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2"
 dropout_5/dropout/GreaterEqual/yц
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
dropout_5/dropout/GreaterEqual
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/dropout/CastЂ
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/dropout/Mul_1Ј
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_16/MatMul/ReadVariableOpЃ
dense_16/MatMulMatMuldropout_5/dropout/Mul_1:z:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_16/MatMulЇ
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_16/BiasAdd/ReadVariableOpЅ
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_16/BiasAdd|
dense_16/SigmoidSigmoiddense_16/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_16/SigmoidЈ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_17/MatMul/ReadVariableOp
dense_17/MatMulMatMuldense_16/Sigmoid:y:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_17/MatMulЇ
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOpЅ
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_17/Softmaxя
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulщ
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulД
IdentityIdentitydense_17/Softmax:softmax:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp#^lstm_5/lstm_cell_10/ReadVariableOp%^lstm_5/lstm_cell_10/ReadVariableOp_1%^lstm_5/lstm_cell_10/ReadVariableOp_2%^lstm_5/lstm_cell_10/ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp)^lstm_5/lstm_cell_10/split/ReadVariableOp+^lstm_5/lstm_cell_10/split_1/ReadVariableOp^lstm_5/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2H
"lstm_5/lstm_cell_10/ReadVariableOp"lstm_5/lstm_cell_10/ReadVariableOp2L
$lstm_5/lstm_cell_10/ReadVariableOp_1$lstm_5/lstm_cell_10/ReadVariableOp_12L
$lstm_5/lstm_cell_10/ReadVariableOp_2$lstm_5/lstm_cell_10/ReadVariableOp_22L
$lstm_5/lstm_cell_10/ReadVariableOp_3$lstm_5/lstm_cell_10/ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2T
(lstm_5/lstm_cell_10/split/ReadVariableOp(lstm_5/lstm_cell_10/split/ReadVariableOp2X
*lstm_5/lstm_cell_10/split_1/ReadVariableOp*lstm_5/lstm_cell_10/split_1/ReadVariableOp2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs
№	
н
D__inference_dense_16_layer_call_and_return_conditional_losses_188414

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_190850

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
з[
з
B__inference_lstm_5_layer_call_and_return_conditional_losses_187642

inputs
lstm_cell_10_187548
lstm_cell_10_187550
lstm_cell_10_187552
identityЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂ$lstm_cell_10/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ&*
shrink_axis_mask2
strided_slice_2
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_187548lstm_cell_10_187550lstm_cell_10_187552*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_1870992&
$lstm_cell_10/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЃ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_187548lstm_cell_10_187550lstm_cell_10_187552*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_187561*
condR
while_cond_187560*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeб
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_10_187548*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulЩ
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_10_187550*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_10/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ&
 
_user_specified_nameinputs
L
ь
__inference__traced_save_191360
file_prefix.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop#
savev2_iter_read_readvariableop	%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop9
5savev2_lstm_5_lstm_cell_10_kernel_read_readvariableopC
?savev2_lstm_5_lstm_cell_10_recurrent_kernel_read_readvariableop7
3savev2_lstm_5_lstm_cell_10_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop0
,savev2_dense_15_kernel_m_read_readvariableop.
*savev2_dense_15_bias_m_read_readvariableop0
,savev2_dense_16_kernel_m_read_readvariableop.
*savev2_dense_16_bias_m_read_readvariableop0
,savev2_dense_17_kernel_m_read_readvariableop.
*savev2_dense_17_bias_m_read_readvariableop;
7savev2_lstm_5_lstm_cell_10_kernel_m_read_readvariableopE
Asavev2_lstm_5_lstm_cell_10_recurrent_kernel_m_read_readvariableop9
5savev2_lstm_5_lstm_cell_10_bias_m_read_readvariableop0
,savev2_dense_15_kernel_v_read_readvariableop.
*savev2_dense_15_bias_v_read_readvariableop0
,savev2_dense_16_kernel_v_read_readvariableop.
*savev2_dense_16_bias_v_read_readvariableop0
,savev2_dense_17_kernel_v_read_readvariableop.
*savev2_dense_17_bias_v_read_readvariableop;
7savev2_lstm_5_lstm_cell_10_kernel_v_read_readvariableopE
Asavev2_lstm_5_lstm_cell_10_recurrent_kernel_v_read_readvariableop9
5savev2_lstm_5_lstm_cell_10_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*Ў
valueЄBЁ%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesв
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЮ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableopsavev2_iter_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop5savev2_lstm_5_lstm_cell_10_kernel_read_readvariableop?savev2_lstm_5_lstm_cell_10_recurrent_kernel_read_readvariableop3savev2_lstm_5_lstm_cell_10_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop,savev2_dense_15_kernel_m_read_readvariableop*savev2_dense_15_bias_m_read_readvariableop,savev2_dense_16_kernel_m_read_readvariableop*savev2_dense_16_bias_m_read_readvariableop,savev2_dense_17_kernel_m_read_readvariableop*savev2_dense_17_bias_m_read_readvariableop7savev2_lstm_5_lstm_cell_10_kernel_m_read_readvariableopAsavev2_lstm_5_lstm_cell_10_recurrent_kernel_m_read_readvariableop5savev2_lstm_5_lstm_cell_10_bias_m_read_readvariableop,savev2_dense_15_kernel_v_read_readvariableop*savev2_dense_15_bias_v_read_readvariableop,savev2_dense_16_kernel_v_read_readvariableop*savev2_dense_16_bias_v_read_readvariableop,savev2_dense_17_kernel_v_read_readvariableop*savev2_dense_17_bias_v_read_readvariableop7savev2_lstm_5_lstm_cell_10_kernel_v_read_readvariableopAsavev2_lstm_5_lstm_cell_10_recurrent_kernel_v_read_readvariableop5savev2_lstm_5_lstm_cell_10_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
§: :  : :  : : :: : : : : :	&:	 :: : : : :  : :  : : ::	&:	 ::  : :  : : ::	&:	 :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	&:%!

_output_shapes
:	 :!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	&:%!

_output_shapes
:	 :!

_output_shapes	
::$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$  

_output_shapes

: : !

_output_shapes
::%"!

_output_shapes
:	&:%#!

_output_shapes
:	 :!$

_output_shapes	
::%

_output_shapes
: 
х
р	
!__inference__wrapped_model_186803
lstm_5_inputB
>sequential_5_lstm_5_lstm_cell_10_split_readvariableop_resourceD
@sequential_5_lstm_5_lstm_cell_10_split_1_readvariableop_resource<
8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource8
4sequential_5_dense_15_matmul_readvariableop_resource9
5sequential_5_dense_15_biasadd_readvariableop_resource8
4sequential_5_dense_16_matmul_readvariableop_resource9
5sequential_5_dense_16_biasadd_readvariableop_resource8
4sequential_5_dense_17_matmul_readvariableop_resource9
5sequential_5_dense_17_biasadd_readvariableop_resource
identityЂ,sequential_5/dense_15/BiasAdd/ReadVariableOpЂ+sequential_5/dense_15/MatMul/ReadVariableOpЂ,sequential_5/dense_16/BiasAdd/ReadVariableOpЂ+sequential_5/dense_16/MatMul/ReadVariableOpЂ,sequential_5/dense_17/BiasAdd/ReadVariableOpЂ+sequential_5/dense_17/MatMul/ReadVariableOpЂ/sequential_5/lstm_5/lstm_cell_10/ReadVariableOpЂ1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_1Ђ1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_2Ђ1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_3Ђ5sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOpЂ7sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOpЂsequential_5/lstm_5/whiler
sequential_5/lstm_5/ShapeShapelstm_5_input*
T0*
_output_shapes
:2
sequential_5/lstm_5/Shape
'sequential_5/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/lstm_5/strided_slice/stack 
)sequential_5/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_5/strided_slice/stack_1 
)sequential_5/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_5/strided_slice/stack_2к
!sequential_5/lstm_5/strided_sliceStridedSlice"sequential_5/lstm_5/Shape:output:00sequential_5/lstm_5/strided_slice/stack:output:02sequential_5/lstm_5/strided_slice/stack_1:output:02sequential_5/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_5/lstm_5/strided_slice
sequential_5/lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_5/lstm_5/zeros/mul/yМ
sequential_5/lstm_5/zeros/mulMul*sequential_5/lstm_5/strided_slice:output:0(sequential_5/lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_5/zeros/mul
 sequential_5/lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 sequential_5/lstm_5/zeros/Less/yЗ
sequential_5/lstm_5/zeros/LessLess!sequential_5/lstm_5/zeros/mul:z:0)sequential_5/lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_5/zeros/Less
"sequential_5/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_5/lstm_5/zeros/packed/1г
 sequential_5/lstm_5/zeros/packedPack*sequential_5/lstm_5/strided_slice:output:0+sequential_5/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_5/lstm_5/zeros/packed
sequential_5/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_5/lstm_5/zeros/ConstХ
sequential_5/lstm_5/zerosFill)sequential_5/lstm_5/zeros/packed:output:0(sequential_5/lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_5/lstm_5/zeros
!sequential_5/lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_5/lstm_5/zeros_1/mul/yТ
sequential_5/lstm_5/zeros_1/mulMul*sequential_5/lstm_5/strided_slice:output:0*sequential_5/lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_5/zeros_1/mul
"sequential_5/lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_5/lstm_5/zeros_1/Less/yП
 sequential_5/lstm_5/zeros_1/LessLess#sequential_5/lstm_5/zeros_1/mul:z:0+sequential_5/lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_5/zeros_1/Less
$sequential_5/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_5/lstm_5/zeros_1/packed/1й
"sequential_5/lstm_5/zeros_1/packedPack*sequential_5/lstm_5/strided_slice:output:0-sequential_5/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_5/lstm_5/zeros_1/packed
!sequential_5/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_5/lstm_5/zeros_1/ConstЭ
sequential_5/lstm_5/zeros_1Fill+sequential_5/lstm_5/zeros_1/packed:output:0*sequential_5/lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_5/lstm_5/zeros_1
"sequential_5/lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_5/lstm_5/transpose/permМ
sequential_5/lstm_5/transpose	Transposelstm_5_input+sequential_5/lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ&2
sequential_5/lstm_5/transpose
sequential_5/lstm_5/Shape_1Shape!sequential_5/lstm_5/transpose:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_5/Shape_1 
)sequential_5/lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_5/strided_slice_1/stackЄ
+sequential_5/lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_1/stack_1Є
+sequential_5/lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_1/stack_2ц
#sequential_5/lstm_5/strided_slice_1StridedSlice$sequential_5/lstm_5/Shape_1:output:02sequential_5/lstm_5/strided_slice_1/stack:output:04sequential_5/lstm_5/strided_slice_1/stack_1:output:04sequential_5/lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_5/lstm_5/strided_slice_1­
/sequential_5/lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ21
/sequential_5/lstm_5/TensorArrayV2/element_shape
!sequential_5/lstm_5/TensorArrayV2TensorListReserve8sequential_5/lstm_5/TensorArrayV2/element_shape:output:0,sequential_5/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_5/lstm_5/TensorArrayV2ч
Isequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   2K
Isequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
;sequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_5/lstm_5/transpose:y:0Rsequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor 
)sequential_5/lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_5/strided_slice_2/stackЄ
+sequential_5/lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_2/stack_1Є
+sequential_5/lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_2/stack_2є
#sequential_5/lstm_5/strided_slice_2StridedSlice!sequential_5/lstm_5/transpose:y:02sequential_5/lstm_5/strided_slice_2/stack:output:04sequential_5/lstm_5/strided_slice_2/stack_1:output:04sequential_5/lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ&*
shrink_axis_mask2%
#sequential_5/lstm_5/strided_slice_2Р
0sequential_5/lstm_5/lstm_cell_10/ones_like/ShapeShape,sequential_5/lstm_5/strided_slice_2:output:0*
T0*
_output_shapes
:22
0sequential_5/lstm_5/lstm_cell_10/ones_like/ShapeЉ
0sequential_5/lstm_5/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_5/lstm_5/lstm_cell_10/ones_like/Const
*sequential_5/lstm_5/lstm_cell_10/ones_likeFill9sequential_5/lstm_5/lstm_cell_10/ones_like/Shape:output:09sequential_5/lstm_5/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2,
*sequential_5/lstm_5/lstm_cell_10/ones_likeК
2sequential_5/lstm_5/lstm_cell_10/ones_like_1/ShapeShape"sequential_5/lstm_5/zeros:output:0*
T0*
_output_shapes
:24
2sequential_5/lstm_5/lstm_cell_10/ones_like_1/Shape­
2sequential_5/lstm_5/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_5/lstm_5/lstm_cell_10/ones_like_1/Const
,sequential_5/lstm_5/lstm_cell_10/ones_like_1Fill;sequential_5/lstm_5/lstm_cell_10/ones_like_1/Shape:output:0;sequential_5/lstm_5/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2.
,sequential_5/lstm_5/lstm_cell_10/ones_like_1ш
$sequential_5/lstm_5/lstm_cell_10/mulMul,sequential_5/lstm_5/strided_slice_2:output:03sequential_5/lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2&
$sequential_5/lstm_5/lstm_cell_10/mulь
&sequential_5/lstm_5/lstm_cell_10/mul_1Mul,sequential_5/lstm_5/strided_slice_2:output:03sequential_5/lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2(
&sequential_5/lstm_5/lstm_cell_10/mul_1ь
&sequential_5/lstm_5/lstm_cell_10/mul_2Mul,sequential_5/lstm_5/strided_slice_2:output:03sequential_5/lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2(
&sequential_5/lstm_5/lstm_cell_10/mul_2ь
&sequential_5/lstm_5/lstm_cell_10/mul_3Mul,sequential_5/lstm_5/strided_slice_2:output:03sequential_5/lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2(
&sequential_5/lstm_5/lstm_cell_10/mul_3
&sequential_5/lstm_5/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_5/lstm_5/lstm_cell_10/ConstІ
0sequential_5/lstm_5/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential_5/lstm_5/lstm_cell_10/split/split_dimю
5sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOpReadVariableOp>sequential_5_lstm_5_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&*
dtype027
5sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOpЋ
&sequential_5/lstm_5/lstm_cell_10/splitSplit9sequential_5/lstm_5/lstm_cell_10/split/split_dim:output:0=sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2(
&sequential_5/lstm_5/lstm_cell_10/splitщ
'sequential_5/lstm_5/lstm_cell_10/MatMulMatMul(sequential_5/lstm_5/lstm_cell_10/mul:z:0/sequential_5/lstm_5/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_5/lstm_5/lstm_cell_10/MatMulя
)sequential_5/lstm_5/lstm_cell_10/MatMul_1MatMul*sequential_5/lstm_5/lstm_cell_10/mul_1:z:0/sequential_5/lstm_5/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_1я
)sequential_5/lstm_5/lstm_cell_10/MatMul_2MatMul*sequential_5/lstm_5/lstm_cell_10/mul_2:z:0/sequential_5/lstm_5/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_2я
)sequential_5/lstm_5/lstm_cell_10/MatMul_3MatMul*sequential_5/lstm_5/lstm_cell_10/mul_3:z:0/sequential_5/lstm_5/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_3
(sequential_5/lstm_5/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_5/lstm_5/lstm_cell_10/Const_1Њ
2sequential_5/lstm_5/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2sequential_5/lstm_5/lstm_cell_10/split_1/split_dim№
7sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOpReadVariableOp@sequential_5_lstm_5_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOpЃ
(sequential_5/lstm_5/lstm_cell_10/split_1Split;sequential_5/lstm_5/lstm_cell_10/split_1/split_dim:output:0?sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2*
(sequential_5/lstm_5/lstm_cell_10/split_1ї
(sequential_5/lstm_5/lstm_cell_10/BiasAddBiasAdd1sequential_5/lstm_5/lstm_cell_10/MatMul:product:01sequential_5/lstm_5/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_5/lstm_5/lstm_cell_10/BiasAdd§
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_1BiasAdd3sequential_5/lstm_5/lstm_cell_10/MatMul_1:product:01sequential_5/lstm_5/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_1§
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_2BiasAdd3sequential_5/lstm_5/lstm_cell_10/MatMul_2:product:01sequential_5/lstm_5/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_2§
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_3BiasAdd3sequential_5/lstm_5/lstm_cell_10/MatMul_3:product:01sequential_5/lstm_5/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_3ф
&sequential_5/lstm_5/lstm_cell_10/mul_4Mul"sequential_5/lstm_5/zeros:output:05sequential_5/lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_5/lstm_5/lstm_cell_10/mul_4ф
&sequential_5/lstm_5/lstm_cell_10/mul_5Mul"sequential_5/lstm_5/zeros:output:05sequential_5/lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_5/lstm_5/lstm_cell_10/mul_5ф
&sequential_5/lstm_5/lstm_cell_10/mul_6Mul"sequential_5/lstm_5/zeros:output:05sequential_5/lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_5/lstm_5/lstm_cell_10/mul_6ф
&sequential_5/lstm_5/lstm_cell_10/mul_7Mul"sequential_5/lstm_5/zeros:output:05sequential_5/lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_5/lstm_5/lstm_cell_10/mul_7м
/sequential_5/lstm_5/lstm_cell_10/ReadVariableOpReadVariableOp8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype021
/sequential_5/lstm_5/lstm_cell_10/ReadVariableOpН
4sequential_5/lstm_5/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4sequential_5/lstm_5/lstm_cell_10/strided_slice/stackС
6sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        28
6sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_1С
6sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_2Т
.sequential_5/lstm_5/lstm_cell_10/strided_sliceStridedSlice7sequential_5/lstm_5/lstm_cell_10/ReadVariableOp:value:0=sequential_5/lstm_5/lstm_cell_10/strided_slice/stack:output:0?sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_1:output:0?sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask20
.sequential_5/lstm_5/lstm_cell_10/strided_sliceї
)sequential_5/lstm_5/lstm_cell_10/MatMul_4MatMul*sequential_5/lstm_5/lstm_cell_10/mul_4:z:07sequential_5/lstm_5/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_4я
$sequential_5/lstm_5/lstm_cell_10/addAddV21sequential_5/lstm_5/lstm_cell_10/BiasAdd:output:03sequential_5/lstm_5/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2&
$sequential_5/lstm_5/lstm_cell_10/addЛ
(sequential_5/lstm_5/lstm_cell_10/SigmoidSigmoid(sequential_5/lstm_5/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2*
(sequential_5/lstm_5/lstm_cell_10/Sigmoidр
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_1ReadVariableOp8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype023
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_1С
6sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        28
6sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stackХ
8sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_1Х
8sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_2Ю
0sequential_5/lstm_5/lstm_cell_10/strided_slice_1StridedSlice9sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_1:value:0?sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_1:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask22
0sequential_5/lstm_5/lstm_cell_10/strided_slice_1љ
)sequential_5/lstm_5/lstm_cell_10/MatMul_5MatMul*sequential_5/lstm_5/lstm_cell_10/mul_5:z:09sequential_5/lstm_5/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_5ѕ
&sequential_5/lstm_5/lstm_cell_10/add_1AddV23sequential_5/lstm_5/lstm_cell_10/BiasAdd_1:output:03sequential_5/lstm_5/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_5/lstm_5/lstm_cell_10/add_1С
*sequential_5/lstm_5/lstm_cell_10/Sigmoid_1Sigmoid*sequential_5/lstm_5/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_5/lstm_5/lstm_cell_10/Sigmoid_1п
&sequential_5/lstm_5/lstm_cell_10/mul_8Mul.sequential_5/lstm_5/lstm_cell_10/Sigmoid_1:y:0$sequential_5/lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_5/lstm_5/lstm_cell_10/mul_8р
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_2ReadVariableOp8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype023
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_2С
6sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   28
6sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stackХ
8sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_1Х
8sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_2Ю
0sequential_5/lstm_5/lstm_cell_10/strided_slice_2StridedSlice9sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_2:value:0?sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_1:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask22
0sequential_5/lstm_5/lstm_cell_10/strided_slice_2љ
)sequential_5/lstm_5/lstm_cell_10/MatMul_6MatMul*sequential_5/lstm_5/lstm_cell_10/mul_6:z:09sequential_5/lstm_5/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_6ѕ
&sequential_5/lstm_5/lstm_cell_10/add_2AddV23sequential_5/lstm_5/lstm_cell_10/BiasAdd_2:output:03sequential_5/lstm_5/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_5/lstm_5/lstm_cell_10/add_2Д
%sequential_5/lstm_5/lstm_cell_10/TanhTanh*sequential_5/lstm_5/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%sequential_5/lstm_5/lstm_cell_10/Tanhт
&sequential_5/lstm_5/lstm_cell_10/mul_9Mul,sequential_5/lstm_5/lstm_cell_10/Sigmoid:y:0)sequential_5/lstm_5/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_5/lstm_5/lstm_cell_10/mul_9у
&sequential_5/lstm_5/lstm_cell_10/add_3AddV2*sequential_5/lstm_5/lstm_cell_10/mul_8:z:0*sequential_5/lstm_5/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_5/lstm_5/lstm_cell_10/add_3р
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_3ReadVariableOp8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 *
dtype023
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_3С
6sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   28
6sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stackХ
8sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_1Х
8sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_2Ю
0sequential_5/lstm_5/lstm_cell_10/strided_slice_3StridedSlice9sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_3:value:0?sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_1:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask22
0sequential_5/lstm_5/lstm_cell_10/strided_slice_3љ
)sequential_5/lstm_5/lstm_cell_10/MatMul_7MatMul*sequential_5/lstm_5/lstm_cell_10/mul_7:z:09sequential_5/lstm_5/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_7ѕ
&sequential_5/lstm_5/lstm_cell_10/add_4AddV23sequential_5/lstm_5/lstm_cell_10/BiasAdd_3:output:03sequential_5/lstm_5/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2(
&sequential_5/lstm_5/lstm_cell_10/add_4С
*sequential_5/lstm_5/lstm_cell_10/Sigmoid_2Sigmoid*sequential_5/lstm_5/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2,
*sequential_5/lstm_5/lstm_cell_10/Sigmoid_2И
'sequential_5/lstm_5/lstm_cell_10/Tanh_1Tanh*sequential_5/lstm_5/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_5/lstm_5/lstm_cell_10/Tanh_1ш
'sequential_5/lstm_5/lstm_cell_10/mul_10Mul.sequential_5/lstm_5/lstm_cell_10/Sigmoid_2:y:0+sequential_5/lstm_5/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'sequential_5/lstm_5/lstm_cell_10/mul_10З
1sequential_5/lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    23
1sequential_5/lstm_5/TensorArrayV2_1/element_shape
#sequential_5/lstm_5/TensorArrayV2_1TensorListReserve:sequential_5/lstm_5/TensorArrayV2_1/element_shape:output:0,sequential_5/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_5/lstm_5/TensorArrayV2_1v
sequential_5/lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_5/lstm_5/timeЇ
,sequential_5/lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,sequential_5/lstm_5/while/maximum_iterations
&sequential_5/lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_5/lstm_5/while/loop_counter
sequential_5/lstm_5/whileWhile/sequential_5/lstm_5/while/loop_counter:output:05sequential_5/lstm_5/while/maximum_iterations:output:0!sequential_5/lstm_5/time:output:0,sequential_5/lstm_5/TensorArrayV2_1:handle:0"sequential_5/lstm_5/zeros:output:0$sequential_5/lstm_5/zeros_1:output:0,sequential_5/lstm_5/strided_slice_1:output:0Ksequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_5_lstm_5_lstm_cell_10_split_readvariableop_resource@sequential_5_lstm_5_lstm_cell_10_split_1_readvariableop_resource8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *%
_read_only_resource_inputs
	
*1
body)R'
%sequential_5_lstm_5_while_body_186645*1
cond)R'
%sequential_5_lstm_5_while_cond_186644*K
output_shapes:
8: : : : :џџџџџџџџџ :џџџџџџџџџ : : : : : *
parallel_iterations 2
sequential_5/lstm_5/whileн
Dsequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2F
Dsequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeИ
6sequential_5/lstm_5/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_5/lstm_5/while:output:3Msequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype028
6sequential_5/lstm_5/TensorArrayV2Stack/TensorListStackЉ
)sequential_5/lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2+
)sequential_5/lstm_5/strided_slice_3/stackЄ
+sequential_5/lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_5/lstm_5/strided_slice_3/stack_1Є
+sequential_5/lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_3/stack_2
#sequential_5/lstm_5/strided_slice_3StridedSlice?sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:02sequential_5/lstm_5/strided_slice_3/stack:output:04sequential_5/lstm_5/strided_slice_3/stack_1:output:04sequential_5/lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2%
#sequential_5/lstm_5/strided_slice_3Ё
$sequential_5/lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_5/lstm_5/transpose_1/permѕ
sequential_5/lstm_5/transpose_1	Transpose?sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_5/lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2!
sequential_5/lstm_5/transpose_1
sequential_5/lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_5/lstm_5/runtimeЯ
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+sequential_5/dense_15/MatMul/ReadVariableOpл
sequential_5/dense_15/MatMulMatMul,sequential_5/lstm_5/strided_slice_3:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_5/dense_15/MatMulЮ
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_15/BiasAdd/ReadVariableOpй
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_5/dense_15/BiasAddЃ
sequential_5/dense_15/SigmoidSigmoid&sequential_5/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_5/dense_15/SigmoidЃ
sequential_5/dropout_5/IdentityIdentity!sequential_5/dense_15/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
sequential_5/dropout_5/IdentityЯ
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+sequential_5/dense_16/MatMul/ReadVariableOpз
sequential_5/dense_16/MatMulMatMul(sequential_5/dropout_5/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_5/dense_16/MatMulЮ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpй
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_5/dense_16/BiasAddЃ
sequential_5/dense_16/SigmoidSigmoid&sequential_5/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sequential_5/dense_16/SigmoidЯ
+sequential_5/dense_17/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_17_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_5/dense_17/MatMul/ReadVariableOpа
sequential_5/dense_17/MatMulMatMul!sequential_5/dense_16/Sigmoid:y:03sequential_5/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_5/dense_17/MatMulЮ
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOpй
sequential_5/dense_17/BiasAddBiasAdd&sequential_5/dense_17/MatMul:product:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_5/dense_17/BiasAddЃ
sequential_5/dense_17/SoftmaxSoftmax&sequential_5/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_5/dense_17/Softmaxю
IdentityIdentity'sequential_5/dense_17/Softmax:softmax:0-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp,^sequential_5/dense_17/MatMul/ReadVariableOp0^sequential_5/lstm_5/lstm_cell_10/ReadVariableOp2^sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_12^sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_22^sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_36^sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOp8^sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOp^sequential_5/lstm_5/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_17/MatMul/ReadVariableOp+sequential_5/dense_17/MatMul/ReadVariableOp2b
/sequential_5/lstm_5/lstm_cell_10/ReadVariableOp/sequential_5/lstm_5/lstm_cell_10/ReadVariableOp2f
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_11sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_12f
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_21sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_22f
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_31sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_32n
5sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOp5sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOp2r
7sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOp7sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOp26
sequential_5/lstm_5/whilesequential_5/lstm_5/while:Y U
+
_output_shapes
:џџџџџџџџџ&
&
_user_specified_namelstm_5_input
Ћ
У
while_cond_189963
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_189963___redundant_placeholder04
0while_while_cond_189963___redundant_placeholder14
0while_while_cond_189963___redundant_placeholder24
0while_while_cond_189963___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
Ш
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_190855

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

г
%sequential_5_lstm_5_while_cond_186644D
@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counterJ
Fsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations)
%sequential_5_lstm_5_while_placeholder+
'sequential_5_lstm_5_while_placeholder_1+
'sequential_5_lstm_5_while_placeholder_2+
'sequential_5_lstm_5_while_placeholder_3F
Bsequential_5_lstm_5_while_less_sequential_5_lstm_5_strided_slice_1\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_186644___redundant_placeholder0\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_186644___redundant_placeholder1\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_186644___redundant_placeholder2\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_186644___redundant_placeholder3&
"sequential_5_lstm_5_while_identity
д
sequential_5/lstm_5/while/LessLess%sequential_5_lstm_5_while_placeholderBsequential_5_lstm_5_while_less_sequential_5_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_5/lstm_5/while/Less
"sequential_5/lstm_5/while/IdentityIdentity"sequential_5/lstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_5/lstm_5/while/Identity"Q
"sequential_5_lstm_5_while_identity+sequential_5/lstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
ю1
Ш
H__inference_sequential_5_layer_call_and_return_conditional_losses_188551

inputs
lstm_5_188515
lstm_5_188517
lstm_5_188519
dense_15_188522
dense_15_188524
dense_16_188528
dense_16_188530
dense_17_188533
dense_17_188535
identityЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCallЂ!dropout_5/StatefulPartitionedCallЂlstm_5/StatefulPartitionedCallЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp
lstm_5/StatefulPartitionedCallStatefulPartitionedCallinputslstm_5_188515lstm_5_188517lstm_5_188519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1880492 
lstm_5/StatefulPartitionedCallЕ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_188522dense_15_188524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1883572"
 dense_15/StatefulPartitionedCall
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1883852#
!dropout_5/StatefulPartitionedCallИ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_16_188528dense_16_188530*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1884142"
 dense_16/StatefulPartitionedCallЗ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_188533dense_17_188535*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1884412"
 dense_17/StatefulPartitionedCallЫ
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_188515*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulУ
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_188517*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulЇ
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs
Џ
ш
$__inference_signature_wrapper_188679
lstm_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_1868032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:џџџџџџџџџ&
&
_user_specified_namelstm_5_input
ђЇ
Ч

lstm_5_while_body_189222*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0?
;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_07
3lstm_5_while_lstm_cell_10_readvariableop_resource_0
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor;
7lstm_5_while_lstm_cell_10_split_readvariableop_resource=
9lstm_5_while_lstm_cell_10_split_1_readvariableop_resource5
1lstm_5_while_lstm_cell_10_readvariableop_resourceЂ(lstm_5/while/lstm_cell_10/ReadVariableOpЂ*lstm_5/while/lstm_cell_10/ReadVariableOp_1Ђ*lstm_5/while/lstm_cell_10/ReadVariableOp_2Ђ*lstm_5/while/lstm_cell_10/ReadVariableOp_3Ђ.lstm_5/while/lstm_cell_10/split/ReadVariableOpЂ0lstm_5/while/lstm_cell_10/split_1/ReadVariableOpб
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   2@
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape§
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ&*
element_dtype022
0lstm_5/while/TensorArrayV2Read/TensorListGetItemН
)lstm_5/while/lstm_cell_10/ones_like/ShapeShape7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/ones_like/Shape
)lstm_5/while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)lstm_5/while/lstm_cell_10/ones_like/Constь
#lstm_5/while/lstm_cell_10/ones_likeFill2lstm_5/while/lstm_cell_10/ones_like/Shape:output:02lstm_5/while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_5/while/lstm_cell_10/ones_likeЄ
+lstm_5/while/lstm_cell_10/ones_like_1/ShapeShapelstm_5_while_placeholder_2*
T0*
_output_shapes
:2-
+lstm_5/while/lstm_cell_10/ones_like_1/Shape
+lstm_5/while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+lstm_5/while/lstm_cell_10/ones_like_1/Constє
%lstm_5/while/lstm_cell_10/ones_like_1Fill4lstm_5/while/lstm_cell_10/ones_like_1/Shape:output:04lstm_5/while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_5/while/lstm_cell_10/ones_like_1о
lstm_5/while/lstm_cell_10/mulMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/while/lstm_cell_10/mulт
lstm_5/while/lstm_cell_10/mul_1Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2!
lstm_5/while/lstm_cell_10/mul_1т
lstm_5/while/lstm_cell_10/mul_2Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2!
lstm_5/while/lstm_cell_10/mul_2т
lstm_5/while/lstm_cell_10/mul_3Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2!
lstm_5/while/lstm_cell_10/mul_3
lstm_5/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
lstm_5/while/lstm_cell_10/Const
)lstm_5/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)lstm_5/while/lstm_cell_10/split/split_dimл
.lstm_5/while/lstm_cell_10/split/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&*
dtype020
.lstm_5/while/lstm_cell_10/split/ReadVariableOp
lstm_5/while/lstm_cell_10/splitSplit2lstm_5/while/lstm_cell_10/split/split_dim:output:06lstm_5/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2!
lstm_5/while/lstm_cell_10/splitЭ
 lstm_5/while/lstm_cell_10/MatMulMatMul!lstm_5/while/lstm_cell_10/mul:z:0(lstm_5/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_5/while/lstm_cell_10/MatMulг
"lstm_5/while/lstm_cell_10/MatMul_1MatMul#lstm_5/while/lstm_cell_10/mul_1:z:0(lstm_5/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_1г
"lstm_5/while/lstm_cell_10/MatMul_2MatMul#lstm_5/while/lstm_cell_10/mul_2:z:0(lstm_5/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_2г
"lstm_5/while/lstm_cell_10/MatMul_3MatMul#lstm_5/while/lstm_cell_10/mul_3:z:0(lstm_5/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_3
!lstm_5/while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2#
!lstm_5/while/lstm_cell_10/Const_1
+lstm_5/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lstm_5/while/lstm_cell_10/split_1/split_dimн
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype022
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp
!lstm_5/while/lstm_cell_10/split_1Split4lstm_5/while/lstm_cell_10/split_1/split_dim:output:08lstm_5/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2#
!lstm_5/while/lstm_cell_10/split_1л
!lstm_5/while/lstm_cell_10/BiasAddBiasAdd*lstm_5/while/lstm_cell_10/MatMul:product:0*lstm_5/while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_5/while/lstm_cell_10/BiasAddс
#lstm_5/while/lstm_cell_10/BiasAdd_1BiasAdd,lstm_5/while/lstm_cell_10/MatMul_1:product:0*lstm_5/while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/while/lstm_cell_10/BiasAdd_1с
#lstm_5/while/lstm_cell_10/BiasAdd_2BiasAdd,lstm_5/while/lstm_cell_10/MatMul_2:product:0*lstm_5/while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/while/lstm_cell_10/BiasAdd_2с
#lstm_5/while/lstm_cell_10/BiasAdd_3BiasAdd,lstm_5/while/lstm_cell_10/MatMul_3:product:0*lstm_5/while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/while/lstm_cell_10/BiasAdd_3Ч
lstm_5/while/lstm_cell_10/mul_4Mullstm_5_while_placeholder_2.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_4Ч
lstm_5/while/lstm_cell_10/mul_5Mullstm_5_while_placeholder_2.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_5Ч
lstm_5/while/lstm_cell_10/mul_6Mullstm_5_while_placeholder_2.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_6Ч
lstm_5/while/lstm_cell_10/mul_7Mullstm_5_while_placeholder_2.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_7Щ
(lstm_5/while/lstm_cell_10/ReadVariableOpReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(lstm_5/while/lstm_cell_10/ReadVariableOpЏ
-lstm_5/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-lstm_5/while/lstm_cell_10/strided_slice/stackГ
/lstm_5/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/lstm_5/while/lstm_cell_10/strided_slice/stack_1Г
/lstm_5/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/lstm_5/while/lstm_cell_10/strided_slice/stack_2
'lstm_5/while/lstm_cell_10/strided_sliceStridedSlice0lstm_5/while/lstm_cell_10/ReadVariableOp:value:06lstm_5/while/lstm_cell_10/strided_slice/stack:output:08lstm_5/while/lstm_cell_10/strided_slice/stack_1:output:08lstm_5/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2)
'lstm_5/while/lstm_cell_10/strided_sliceл
"lstm_5/while/lstm_cell_10/MatMul_4MatMul#lstm_5/while/lstm_cell_10/mul_4:z:00lstm_5/while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_4г
lstm_5/while/lstm_cell_10/addAddV2*lstm_5/while/lstm_cell_10/BiasAdd:output:0,lstm_5/while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/while/lstm_cell_10/addІ
!lstm_5/while/lstm_cell_10/SigmoidSigmoid!lstm_5/while/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_5/while/lstm_cell_10/SigmoidЭ
*lstm_5/while/lstm_cell_10/ReadVariableOp_1ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_1Г
/lstm_5/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/lstm_5/while/lstm_cell_10/strided_slice_1/stackЗ
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   23
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_1З
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_2Є
)lstm_5/while/lstm_cell_10/strided_slice_1StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_1:value:08lstm_5/while/lstm_cell_10/strided_slice_1/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_1/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_1н
"lstm_5/while/lstm_cell_10/MatMul_5MatMul#lstm_5/while/lstm_cell_10/mul_5:z:02lstm_5/while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_5й
lstm_5/while/lstm_cell_10/add_1AddV2,lstm_5/while/lstm_cell_10/BiasAdd_1:output:0,lstm_5/while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/add_1Ќ
#lstm_5/while/lstm_cell_10/Sigmoid_1Sigmoid#lstm_5/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/while/lstm_cell_10/Sigmoid_1Р
lstm_5/while/lstm_cell_10/mul_8Mul'lstm_5/while/lstm_cell_10/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_8Э
*lstm_5/while/lstm_cell_10/ReadVariableOp_2ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_2Г
/lstm_5/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   21
/lstm_5/while/lstm_cell_10/strided_slice_2/stackЗ
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   23
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_1З
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_2Є
)lstm_5/while/lstm_cell_10/strided_slice_2StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_2:value:08lstm_5/while/lstm_cell_10/strided_slice_2/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_2/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_2н
"lstm_5/while/lstm_cell_10/MatMul_6MatMul#lstm_5/while/lstm_cell_10/mul_6:z:02lstm_5/while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_6й
lstm_5/while/lstm_cell_10/add_2AddV2,lstm_5/while/lstm_cell_10/BiasAdd_2:output:0,lstm_5/while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/add_2
lstm_5/while/lstm_cell_10/TanhTanh#lstm_5/while/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_5/while/lstm_cell_10/TanhЦ
lstm_5/while/lstm_cell_10/mul_9Mul%lstm_5/while/lstm_cell_10/Sigmoid:y:0"lstm_5/while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_9Ч
lstm_5/while/lstm_cell_10/add_3AddV2#lstm_5/while/lstm_cell_10/mul_8:z:0#lstm_5/while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/add_3Э
*lstm_5/while/lstm_cell_10/ReadVariableOp_3ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_3Г
/lstm_5/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   21
/lstm_5/while/lstm_cell_10/strided_slice_3/stackЗ
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_1З
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_2Є
)lstm_5/while/lstm_cell_10/strided_slice_3StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_3:value:08lstm_5/while/lstm_cell_10/strided_slice_3/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_3/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_3н
"lstm_5/while/lstm_cell_10/MatMul_7MatMul#lstm_5/while/lstm_cell_10/mul_7:z:02lstm_5/while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_7й
lstm_5/while/lstm_cell_10/add_4AddV2,lstm_5/while/lstm_cell_10/BiasAdd_3:output:0,lstm_5/while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/add_4Ќ
#lstm_5/while/lstm_cell_10/Sigmoid_2Sigmoid#lstm_5/while/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/while/lstm_cell_10/Sigmoid_2Ѓ
 lstm_5/while/lstm_cell_10/Tanh_1Tanh#lstm_5/while/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_5/while/lstm_cell_10/Tanh_1Ь
 lstm_5/while/lstm_cell_10/mul_10Mul'lstm_5/while/lstm_cell_10/Sigmoid_2:y:0$lstm_5/while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_5/while/lstm_cell_10/mul_10
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder$lstm_5/while/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype023
1lstm_5/while/TensorArrayV2Write/TensorListSetItemj
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add/y
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/addn
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add_1/y
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/add_1
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/IdentityЃ
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_1
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_2И
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_3Ќ
lstm_5/while/Identity_4Identity$lstm_5/while/lstm_cell_10/mul_10:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/while/Identity_4Ћ
lstm_5/while/Identity_5Identity#lstm_5/while/lstm_cell_10/add_3:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/while/Identity_5"7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"h
1lstm_5_while_lstm_cell_10_readvariableop_resource3lstm_5_while_lstm_cell_10_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_10_split_1_readvariableop_resource;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_10_split_readvariableop_resource9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0"Ф
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ :џџџџџџџџџ : : :::2T
(lstm_5/while/lstm_cell_10/ReadVariableOp(lstm_5/while/lstm_cell_10/ReadVariableOp2X
*lstm_5/while/lstm_cell_10/ReadVariableOp_1*lstm_5/while/lstm_cell_10/ReadVariableOp_12X
*lstm_5/while/lstm_cell_10/ReadVariableOp_2*lstm_5/while/lstm_cell_10/ReadVariableOp_22X
*lstm_5/while/lstm_cell_10/ReadVariableOp_3*lstm_5/while/lstm_cell_10/ReadVariableOp_32`
.lstm_5/while/lstm_cell_10/split/ReadVariableOp.lstm_5/while/lstm_cell_10/split/ReadVariableOp2d
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_188385

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

в
while_body_189964
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resourceЂ!while/lstm_cell_10/ReadVariableOpЂ#while/lstm_cell_10/ReadVariableOp_1Ђ#while/lstm_cell_10/ReadVariableOp_2Ђ#while/lstm_cell_10/ReadVariableOp_3Ђ'while/lstm_cell_10/split/ReadVariableOpЂ)while/lstm_cell_10/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЈ
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_10/ones_like/Constа
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/ones_like
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$while/lstm_cell_10/ones_like_1/Constи
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_10/ones_like_1Т
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mulЦ
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_1Ц
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_2Ц
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЦ
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpѓ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/splitБ
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMulЗ
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_1З
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_2З
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dimШ
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpы
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1П
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAddХ
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_1Х
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_2Х
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_3Ћ
while/lstm_cell_10/mul_4Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_4Ћ
while/lstm_cell_10/mul_5Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_5Ћ
while/lstm_cell_10/mul_6Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_6Ћ
while/lstm_cell_10/mul_7Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_7Д
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_10/ReadVariableOpЁ
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stackЅ
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1Ѕ
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2ю
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceП
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_4З
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/SigmoidИ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_1Ѕ
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stackЉ
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1Љ
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2њ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1С
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_5Н
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_1Є
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_8И
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_2Ѕ
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stackЉ
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1Љ
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2њ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2С
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_6Н
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_2
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/TanhЊ
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_9Ћ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_3И
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_3Ѕ
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stackЉ
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1Љ
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2њ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3С
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_7Н
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_4
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Tanh_1А
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_10с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3э
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4ь
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ :џџџџџџџџџ : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
ї
Е
__inference_loss_fn_1_191229G
Clstm_5_lstm_cell_10_bias_regularizer_square_readvariableop_resource
identityЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpљ
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpClstm_5_lstm_cell_10_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulЌ
IdentityIdentity,lstm_5/lstm_cell_10/bias/Regularizer/mul:z:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp


'__inference_lstm_5_layer_call_fn_190818
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1876422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ&:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ&
"
_user_specified_name
inputs/0
Ћ
У
while_cond_190316
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_190316___redundant_placeholder04
0while_while_cond_190316___redundant_placeholder14
0while_while_cond_190316___redundant_placeholder24
0while_while_cond_190316___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
Э
ы
-__inference_sequential_5_layer_call_fn_189438

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1886132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs
Ж
Й
__inference_loss_fn_0_191218I
Elstm_5_lstm_cell_10_kernel_regularizer_square_readvariableop_resource
identityЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpElstm_5_lstm_cell_10_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulА
IdentityIdentity.lstm_5/lstm_cell_10/kernel/Regularizer/mul:z:0=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp
ђј
в
while_body_190317
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resourceЂ!while/lstm_cell_10/ReadVariableOpЂ#while/lstm_cell_10/ReadVariableOp_1Ђ#while/lstm_cell_10/ReadVariableOp_2Ђ#while/lstm_cell_10/ReadVariableOp_3Ђ'while/lstm_cell_10/split/ReadVariableOpЂ)while/lstm_cell_10/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЈ
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_10/ones_like/Constа
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/ones_like
 while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 while/lstm_cell_10/dropout/ConstЫ
while/lstm_cell_10/dropout/MulMul%while/lstm_cell_10/ones_like:output:0)while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2 
while/lstm_cell_10/dropout/Mul
 while/lstm_cell_10/dropout/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_10/dropout/Shape
7while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2уЧ29
7while/lstm_cell_10/dropout/random_uniform/RandomUniform
)while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2+
)while/lstm_cell_10/dropout/GreaterEqual/y
'while/lstm_cell_10/dropout/GreaterEqualGreaterEqual@while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2)
'while/lstm_cell_10/dropout/GreaterEqualИ
while/lstm_cell_10/dropout/CastCast+while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2!
while/lstm_cell_10/dropout/CastЦ
 while/lstm_cell_10/dropout/Mul_1Mul"while/lstm_cell_10/dropout/Mul:z:0#while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout/Mul_1
"while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_1/Constб
 while/lstm_cell_10/dropout_1/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout_1/Mul
"while/lstm_cell_10/dropout_1/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_1/Shape
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ЩФж2;
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_1/GreaterEqual/y
)while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)while/lstm_cell_10/dropout_1/GreaterEqualО
!while/lstm_cell_10/dropout_1/CastCast-while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2#
!while/lstm_cell_10/dropout_1/CastЮ
"while/lstm_cell_10/dropout_1/Mul_1Mul$while/lstm_cell_10/dropout_1/Mul:z:0%while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2$
"while/lstm_cell_10/dropout_1/Mul_1
"while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_2/Constб
 while/lstm_cell_10/dropout_2/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout_2/Mul
"while/lstm_cell_10/dropout_2/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_2/Shape
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ЃЉИ2;
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_2/GreaterEqual/y
)while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)while/lstm_cell_10/dropout_2/GreaterEqualО
!while/lstm_cell_10/dropout_2/CastCast-while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2#
!while/lstm_cell_10/dropout_2/CastЮ
"while/lstm_cell_10/dropout_2/Mul_1Mul$while/lstm_cell_10/dropout_2/Mul:z:0%while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2$
"while/lstm_cell_10/dropout_2/Mul_1
"while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_3/Constб
 while/lstm_cell_10/dropout_3/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout_3/Mul
"while/lstm_cell_10/dropout_3/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_3/Shape
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ќк2;
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_3/GreaterEqual/y
)while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)while/lstm_cell_10/dropout_3/GreaterEqualО
!while/lstm_cell_10/dropout_3/CastCast-while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2#
!while/lstm_cell_10/dropout_3/CastЮ
"while/lstm_cell_10/dropout_3/Mul_1Mul$while/lstm_cell_10/dropout_3/Mul:z:0%while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2$
"while/lstm_cell_10/dropout_3/Mul_1
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$while/lstm_cell_10/ones_like_1/Constи
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_10/ones_like_1
"while/lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_4/Constг
 while/lstm_cell_10/dropout_4/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_4/Mul
"while/lstm_cell_10/dropout_4/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_4/Shape
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2эЭ§2;
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_4/GreaterEqual/y
)while/lstm_cell_10/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_4/GreaterEqualО
!while/lstm_cell_10/dropout_4/CastCast-while/lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_4/CastЮ
"while/lstm_cell_10/dropout_4/Mul_1Mul$while/lstm_cell_10/dropout_4/Mul:z:0%while/lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_4/Mul_1
"while/lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_5/Constг
 while/lstm_cell_10/dropout_5/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_5/Mul
"while/lstm_cell_10/dropout_5/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_5/Shape
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2оя2;
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_5/GreaterEqual/y
)while/lstm_cell_10/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_5/GreaterEqualО
!while/lstm_cell_10/dropout_5/CastCast-while/lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_5/CastЮ
"while/lstm_cell_10/dropout_5/Mul_1Mul$while/lstm_cell_10/dropout_5/Mul:z:0%while/lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_5/Mul_1
"while/lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_6/Constг
 while/lstm_cell_10/dropout_6/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_6/Mul
"while/lstm_cell_10/dropout_6/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_6/Shape
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Љй2;
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_6/GreaterEqual/y
)while/lstm_cell_10/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_6/GreaterEqualО
!while/lstm_cell_10/dropout_6/CastCast-while/lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_6/CastЮ
"while/lstm_cell_10/dropout_6/Mul_1Mul$while/lstm_cell_10/dropout_6/Mul:z:0%while/lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_6/Mul_1
"while/lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_7/Constг
 while/lstm_cell_10/dropout_7/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_7/Mul
"while/lstm_cell_10/dropout_7/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_7/Shape
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЏЛ2;
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_7/GreaterEqual/y
)while/lstm_cell_10/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_7/GreaterEqualО
!while/lstm_cell_10/dropout_7/CastCast-while/lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_7/CastЮ
"while/lstm_cell_10/dropout_7/Mul_1Mul$while/lstm_cell_10/dropout_7/Mul:z:0%while/lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_7/Mul_1С
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mulЧ
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_1Ч
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_2Ч
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЦ
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpѓ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/splitБ
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMulЗ
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_1З
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_2З
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dimШ
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpы
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1П
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAddХ
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_1Х
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_2Х
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_3Њ
while/lstm_cell_10/mul_4Mulwhile_placeholder_2&while/lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_4Њ
while/lstm_cell_10/mul_5Mulwhile_placeholder_2&while/lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_5Њ
while/lstm_cell_10/mul_6Mulwhile_placeholder_2&while/lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_6Њ
while/lstm_cell_10/mul_7Mulwhile_placeholder_2&while/lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_7Д
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_10/ReadVariableOpЁ
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stackЅ
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1Ѕ
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2ю
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceП
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_4З
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/SigmoidИ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_1Ѕ
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stackЉ
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1Љ
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2њ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1С
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_5Н
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_1Є
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_8И
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_2Ѕ
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stackЉ
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1Љ
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2њ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2С
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_6Н
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_2
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/TanhЊ
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_9Ћ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_3И
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_3Ѕ
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stackЉ
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1Љ
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2њ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3С
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_7Н
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_4
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Tanh_1А
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_10с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3э
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4ь
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ :џџџџџџџџџ : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
ў	
Я
lstm_5_while_cond_189221*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1B
>lstm_5_while_lstm_5_while_cond_189221___redundant_placeholder0B
>lstm_5_while_lstm_5_while_cond_189221___redundant_placeholder1B
>lstm_5_while_lstm_5_while_cond_189221___redundant_placeholder2B
>lstm_5_while_lstm_5_while_cond_189221___redundant_placeholder3
lstm_5_while_identity

lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
lstm_5/while/Lessr
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_5/while/Identity"7
lstm_5_while_identitylstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ :џџџџџџџџџ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
:
О0
Є
H__inference_sequential_5_layer_call_and_return_conditional_losses_188613

inputs
lstm_5_188577
lstm_5_188579
lstm_5_188581
dense_15_188584
dense_15_188586
dense_16_188590
dense_16_188592
dense_17_188595
dense_17_188597
identityЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCallЂlstm_5/StatefulPartitionedCallЂ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp
lstm_5/StatefulPartitionedCallStatefulPartitionedCallinputslstm_5_188577lstm_5_188579lstm_5_188581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1883162 
lstm_5/StatefulPartitionedCallЕ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_188584dense_15_188586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1883572"
 dense_15/StatefulPartitionedCallњ
dropout_5/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1883902
dropout_5/PartitionedCallА
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_16_188590dense_16_188592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1884142"
 dense_16/StatefulPartitionedCallЗ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_188595dense_17_188597*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1884412"
 dense_17/StatefulPartitionedCallЫ
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_188577*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulУ
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_188579*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs


'__inference_lstm_5_layer_call_fn_190807
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_1874982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ&:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ&
"
_user_specified_name
inputs/0

в
while_body_190648
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resourceЂ!while/lstm_cell_10/ReadVariableOpЂ#while/lstm_cell_10/ReadVariableOp_1Ђ#while/lstm_cell_10/ReadVariableOp_2Ђ#while/lstm_cell_10/ReadVariableOp_3Ђ'while/lstm_cell_10/split/ReadVariableOpЂ)while/lstm_cell_10/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЈ
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_10/ones_like/Constа
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/ones_like
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$while/lstm_cell_10/ones_like_1/Constи
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_10/ones_like_1Т
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mulЦ
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_1Ц
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_2Ц
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЦ
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpѓ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/splitБ
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMulЗ
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_1З
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_2З
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dimШ
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpы
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1П
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAddХ
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_1Х
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_2Х
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_3Ћ
while/lstm_cell_10/mul_4Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_4Ћ
while/lstm_cell_10/mul_5Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_5Ћ
while/lstm_cell_10/mul_6Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_6Ћ
while/lstm_cell_10/mul_7Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_7Д
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_10/ReadVariableOpЁ
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stackЅ
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1Ѕ
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2ю
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceП
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_4З
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/SigmoidИ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_1Ѕ
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stackЉ
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1Љ
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2њ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1С
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_5Н
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_1Є
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_8И
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_2Ѕ
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stackЉ
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1Љ
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2њ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2С
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_6Н
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_2
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/TanhЊ
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_9Ћ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_3И
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_3Ѕ
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stackЉ
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1Љ
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2њ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3С
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_7Н
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_4
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Tanh_1А
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_10с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3э
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4ь
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ :џџџџџџџџџ : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
м
~
)__inference_dense_16_layer_call_fn_190885

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1884142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
№	
н
D__inference_dense_16_layer_call_and_return_conditional_losses_190876

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

в
while_body_188168
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resourceЂ!while/lstm_cell_10/ReadVariableOpЂ#while/lstm_cell_10/ReadVariableOp_1Ђ#while/lstm_cell_10/ReadVariableOp_2Ђ#while/lstm_cell_10/ReadVariableOp_3Ђ'while/lstm_cell_10/split/ReadVariableOpЂ)while/lstm_cell_10/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЈ
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_10/ones_like/Constа
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/ones_like
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$while/lstm_cell_10/ones_like_1/Constи
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_10/ones_like_1Т
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mulЦ
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_1Ц
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_2Ц
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЦ
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpѓ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/splitБ
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMulЗ
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_1З
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_2З
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dimШ
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpы
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1П
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAddХ
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_1Х
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_2Х
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_3Ћ
while/lstm_cell_10/mul_4Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_4Ћ
while/lstm_cell_10/mul_5Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_5Ћ
while/lstm_cell_10/mul_6Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_6Ћ
while/lstm_cell_10/mul_7Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_7Д
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_10/ReadVariableOpЁ
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stackЅ
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1Ѕ
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2ю
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceП
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_4З
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/SigmoidИ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_1Ѕ
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stackЉ
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1Љ
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2њ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1С
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_5Н
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_1Є
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_8И
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_2Ѕ
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stackЉ
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1Љ
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2њ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2С
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_6Н
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_2
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/TanhЊ
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_9Ћ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_3И
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_3Ѕ
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stackЉ
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1Љ
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2њ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3С
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_7Н
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_4
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Tanh_1А
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_10с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3э
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4ь
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ :џџџџџџџџџ : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
ЙЏ
ќ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_187003

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3Ђ:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЂ<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpЂsplit/ReadVariableOpЂsplit_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeв
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ЄБ22&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeй
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2џЫ2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_1/GreaterEqual/yЦ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeй
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2њ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_2/GreaterEqual/yЦ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeй
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2§Юо2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_3/GreaterEqual/yЦ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
dropout_3/Mul_1\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_4/Const
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shapeй
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Љ2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_4/GreaterEqual/yЦ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_5/Const
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shapeй
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ыЗ2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_5/GreaterEqual/yЦ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_6/Const
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shapeй
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЦЃЯ2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_6/GreaterEqual/yЦ
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_6/GreaterEqual
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_6/Cast
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_7/Const
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shapeй
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2лт2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_7/GreaterEqual/yЦ
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_7/GreaterEqual
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2
dropout_7/Cast
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
	BiasAdd_3d
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_4d
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_5d
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_6d
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_10л
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpи
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square­
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Constъ
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/SumЁ
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xь
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mulе
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpЮ
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+lstm_5/lstm_cell_10/bias/Regularizer/SquareЂ
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Constт
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xф
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mulд
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityи

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1з

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ&:џџџџџџџџџ :џџџџџџџџџ :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_namestates
Л
Э
-__inference_lstm_cell_10_layer_call_fn_191190

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_1870032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ&:џџџџџџџџџ :џџџџџџџџџ :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ&
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
states/1

F
*__inference_dropout_5_layer_call_fn_190865

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1883902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
й
Ч

lstm_5_while_body_188862*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0?
;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_07
3lstm_5_while_lstm_cell_10_readvariableop_resource_0
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor;
7lstm_5_while_lstm_cell_10_split_readvariableop_resource=
9lstm_5_while_lstm_cell_10_split_1_readvariableop_resource5
1lstm_5_while_lstm_cell_10_readvariableop_resourceЂ(lstm_5/while/lstm_cell_10/ReadVariableOpЂ*lstm_5/while/lstm_cell_10/ReadVariableOp_1Ђ*lstm_5/while/lstm_cell_10/ReadVariableOp_2Ђ*lstm_5/while/lstm_cell_10/ReadVariableOp_3Ђ.lstm_5/while/lstm_cell_10/split/ReadVariableOpЂ0lstm_5/while/lstm_cell_10/split_1/ReadVariableOpб
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   2@
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape§
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ&*
element_dtype022
0lstm_5/while/TensorArrayV2Read/TensorListGetItemН
)lstm_5/while/lstm_cell_10/ones_like/ShapeShape7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/ones_like/Shape
)lstm_5/while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)lstm_5/while/lstm_cell_10/ones_like/Constь
#lstm_5/while/lstm_cell_10/ones_likeFill2lstm_5/while/lstm_cell_10/ones_like/Shape:output:02lstm_5/while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2%
#lstm_5/while/lstm_cell_10/ones_like
'lstm_5/while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'lstm_5/while/lstm_cell_10/dropout/Constч
%lstm_5/while/lstm_cell_10/dropout/MulMul,lstm_5/while/lstm_cell_10/ones_like:output:00lstm_5/while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2'
%lstm_5/while/lstm_cell_10/dropout/MulЎ
'lstm_5/while/lstm_cell_10/dropout/ShapeShape,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2)
'lstm_5/while/lstm_cell_10/dropout/ShapeЁ
>lstm_5/while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform0lstm_5/while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ЉУ2@
>lstm_5/while/lstm_cell_10/dropout/random_uniform/RandomUniformЉ
0lstm_5/while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>22
0lstm_5/while/lstm_cell_10/dropout/GreaterEqual/yІ
.lstm_5/while/lstm_cell_10/dropout/GreaterEqualGreaterEqualGlstm_5/while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:09lstm_5/while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&20
.lstm_5/while/lstm_cell_10/dropout/GreaterEqualЭ
&lstm_5/while/lstm_cell_10/dropout/CastCast2lstm_5/while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2(
&lstm_5/while/lstm_cell_10/dropout/Castт
'lstm_5/while/lstm_cell_10/dropout/Mul_1Mul)lstm_5/while/lstm_cell_10/dropout/Mul:z:0*lstm_5/while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2)
'lstm_5/while/lstm_cell_10/dropout/Mul_1
)lstm_5/while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)lstm_5/while/lstm_cell_10/dropout_1/Constэ
'lstm_5/while/lstm_cell_10/dropout_1/MulMul,lstm_5/while/lstm_cell_10/ones_like:output:02lstm_5/while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2)
'lstm_5/while/lstm_cell_10/dropout_1/MulВ
)lstm_5/while/lstm_cell_10/dropout_1/ShapeShape,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_1/ShapeЇ
@lstm_5/while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ЋТ2B
@lstm_5/while/lstm_cell_10/dropout_1/random_uniform/RandomUniform­
2lstm_5/while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>24
2lstm_5/while/lstm_cell_10/dropout_1/GreaterEqual/yЎ
0lstm_5/while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&22
0lstm_5/while/lstm_cell_10/dropout_1/GreaterEqualг
(lstm_5/while/lstm_cell_10/dropout_1/CastCast4lstm_5/while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2*
(lstm_5/while/lstm_cell_10/dropout_1/Castъ
)lstm_5/while/lstm_cell_10/dropout_1/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_1/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)lstm_5/while/lstm_cell_10/dropout_1/Mul_1
)lstm_5/while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)lstm_5/while/lstm_cell_10/dropout_2/Constэ
'lstm_5/while/lstm_cell_10/dropout_2/MulMul,lstm_5/while/lstm_cell_10/ones_like:output:02lstm_5/while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2)
'lstm_5/while/lstm_cell_10/dropout_2/MulВ
)lstm_5/while/lstm_cell_10/dropout_2/ShapeShape,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_2/ShapeЇ
@lstm_5/while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2 2B
@lstm_5/while/lstm_cell_10/dropout_2/random_uniform/RandomUniform­
2lstm_5/while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>24
2lstm_5/while/lstm_cell_10/dropout_2/GreaterEqual/yЎ
0lstm_5/while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&22
0lstm_5/while/lstm_cell_10/dropout_2/GreaterEqualг
(lstm_5/while/lstm_cell_10/dropout_2/CastCast4lstm_5/while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2*
(lstm_5/while/lstm_cell_10/dropout_2/Castъ
)lstm_5/while/lstm_cell_10/dropout_2/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_2/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)lstm_5/while/lstm_cell_10/dropout_2/Mul_1
)lstm_5/while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)lstm_5/while/lstm_cell_10/dropout_3/Constэ
'lstm_5/while/lstm_cell_10/dropout_3/MulMul,lstm_5/while/lstm_cell_10/ones_like:output:02lstm_5/while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2)
'lstm_5/while/lstm_cell_10/dropout_3/MulВ
)lstm_5/while/lstm_cell_10/dropout_3/ShapeShape,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_3/ShapeЇ
@lstm_5/while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ЙПЩ2B
@lstm_5/while/lstm_cell_10/dropout_3/random_uniform/RandomUniform­
2lstm_5/while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>24
2lstm_5/while/lstm_cell_10/dropout_3/GreaterEqual/yЎ
0lstm_5/while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&22
0lstm_5/while/lstm_cell_10/dropout_3/GreaterEqualг
(lstm_5/while/lstm_cell_10/dropout_3/CastCast4lstm_5/while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2*
(lstm_5/while/lstm_cell_10/dropout_3/Castъ
)lstm_5/while/lstm_cell_10/dropout_3/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_3/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)lstm_5/while/lstm_cell_10/dropout_3/Mul_1Є
+lstm_5/while/lstm_cell_10/ones_like_1/ShapeShapelstm_5_while_placeholder_2*
T0*
_output_shapes
:2-
+lstm_5/while/lstm_cell_10/ones_like_1/Shape
+lstm_5/while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+lstm_5/while/lstm_cell_10/ones_like_1/Constє
%lstm_5/while/lstm_cell_10/ones_like_1Fill4lstm_5/while/lstm_cell_10/ones_like_1/Shape:output:04lstm_5/while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2'
%lstm_5/while/lstm_cell_10/ones_like_1
)lstm_5/while/lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)lstm_5/while/lstm_cell_10/dropout_4/Constя
'lstm_5/while/lstm_cell_10/dropout_4/MulMul.lstm_5/while/lstm_cell_10/ones_like_1:output:02lstm_5/while/lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'lstm_5/while/lstm_cell_10/dropout_4/MulД
)lstm_5/while/lstm_cell_10/dropout_4/ShapeShape.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_4/ShapeЇ
@lstm_5/while/lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЌВЩ2B
@lstm_5/while/lstm_cell_10/dropout_4/random_uniform/RandomUniform­
2lstm_5/while/lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>24
2lstm_5/while/lstm_cell_10/dropout_4/GreaterEqual/yЎ
0lstm_5/while/lstm_cell_10/dropout_4/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0lstm_5/while/lstm_cell_10/dropout_4/GreaterEqualг
(lstm_5/while/lstm_cell_10/dropout_4/CastCast4lstm_5/while/lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_5/while/lstm_cell_10/dropout_4/Castъ
)lstm_5/while/lstm_cell_10/dropout_4/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_4/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_5/while/lstm_cell_10/dropout_4/Mul_1
)lstm_5/while/lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)lstm_5/while/lstm_cell_10/dropout_5/Constя
'lstm_5/while/lstm_cell_10/dropout_5/MulMul.lstm_5/while/lstm_cell_10/ones_like_1:output:02lstm_5/while/lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'lstm_5/while/lstm_cell_10/dropout_5/MulД
)lstm_5/while/lstm_cell_10/dropout_5/ShapeShape.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_5/ShapeІ
@lstm_5/while/lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ИЂ2B
@lstm_5/while/lstm_cell_10/dropout_5/random_uniform/RandomUniform­
2lstm_5/while/lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>24
2lstm_5/while/lstm_cell_10/dropout_5/GreaterEqual/yЎ
0lstm_5/while/lstm_cell_10/dropout_5/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0lstm_5/while/lstm_cell_10/dropout_5/GreaterEqualг
(lstm_5/while/lstm_cell_10/dropout_5/CastCast4lstm_5/while/lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_5/while/lstm_cell_10/dropout_5/Castъ
)lstm_5/while/lstm_cell_10/dropout_5/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_5/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_5/while/lstm_cell_10/dropout_5/Mul_1
)lstm_5/while/lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)lstm_5/while/lstm_cell_10/dropout_6/Constя
'lstm_5/while/lstm_cell_10/dropout_6/MulMul.lstm_5/while/lstm_cell_10/ones_like_1:output:02lstm_5/while/lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'lstm_5/while/lstm_cell_10/dropout_6/MulД
)lstm_5/while/lstm_cell_10/dropout_6/ShapeShape.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_6/ShapeЇ
@lstm_5/while/lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2Ц2B
@lstm_5/while/lstm_cell_10/dropout_6/random_uniform/RandomUniform­
2lstm_5/while/lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>24
2lstm_5/while/lstm_cell_10/dropout_6/GreaterEqual/yЎ
0lstm_5/while/lstm_cell_10/dropout_6/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0lstm_5/while/lstm_cell_10/dropout_6/GreaterEqualг
(lstm_5/while/lstm_cell_10/dropout_6/CastCast4lstm_5/while/lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_5/while/lstm_cell_10/dropout_6/Castъ
)lstm_5/while/lstm_cell_10/dropout_6/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_6/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_5/while/lstm_cell_10/dropout_6/Mul_1
)lstm_5/while/lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)lstm_5/while/lstm_cell_10/dropout_7/Constя
'lstm_5/while/lstm_cell_10/dropout_7/MulMul.lstm_5/while/lstm_cell_10/ones_like_1:output:02lstm_5/while/lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2)
'lstm_5/while/lstm_cell_10/dropout_7/MulД
)lstm_5/while/lstm_cell_10/dropout_7/ShapeShape.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_7/ShapeЇ
@lstm_5/while/lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ЩТ2B
@lstm_5/while/lstm_cell_10/dropout_7/random_uniform/RandomUniform­
2lstm_5/while/lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>24
2lstm_5/while/lstm_cell_10/dropout_7/GreaterEqual/yЎ
0lstm_5/while/lstm_cell_10/dropout_7/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 22
0lstm_5/while/lstm_cell_10/dropout_7/GreaterEqualг
(lstm_5/while/lstm_cell_10/dropout_7/CastCast4lstm_5/while/lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2*
(lstm_5/while/lstm_cell_10/dropout_7/Castъ
)lstm_5/while/lstm_cell_10/dropout_7/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_7/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)lstm_5/while/lstm_cell_10/dropout_7/Mul_1н
lstm_5/while/lstm_cell_10/mulMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_5/while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
lstm_5/while/lstm_cell_10/mulу
lstm_5/while/lstm_cell_10/mul_1Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_5/while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2!
lstm_5/while/lstm_cell_10/mul_1у
lstm_5/while/lstm_cell_10/mul_2Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_5/while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2!
lstm_5/while/lstm_cell_10/mul_2у
lstm_5/while/lstm_cell_10/mul_3Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_5/while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2!
lstm_5/while/lstm_cell_10/mul_3
lstm_5/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
lstm_5/while/lstm_cell_10/Const
)lstm_5/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)lstm_5/while/lstm_cell_10/split/split_dimл
.lstm_5/while/lstm_cell_10/split/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&*
dtype020
.lstm_5/while/lstm_cell_10/split/ReadVariableOp
lstm_5/while/lstm_cell_10/splitSplit2lstm_5/while/lstm_cell_10/split/split_dim:output:06lstm_5/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2!
lstm_5/while/lstm_cell_10/splitЭ
 lstm_5/while/lstm_cell_10/MatMulMatMul!lstm_5/while/lstm_cell_10/mul:z:0(lstm_5/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_5/while/lstm_cell_10/MatMulг
"lstm_5/while/lstm_cell_10/MatMul_1MatMul#lstm_5/while/lstm_cell_10/mul_1:z:0(lstm_5/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_1г
"lstm_5/while/lstm_cell_10/MatMul_2MatMul#lstm_5/while/lstm_cell_10/mul_2:z:0(lstm_5/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_2г
"lstm_5/while/lstm_cell_10/MatMul_3MatMul#lstm_5/while/lstm_cell_10/mul_3:z:0(lstm_5/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_3
!lstm_5/while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2#
!lstm_5/while/lstm_cell_10/Const_1
+lstm_5/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lstm_5/while/lstm_cell_10/split_1/split_dimн
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype022
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp
!lstm_5/while/lstm_cell_10/split_1Split4lstm_5/while/lstm_cell_10/split_1/split_dim:output:08lstm_5/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2#
!lstm_5/while/lstm_cell_10/split_1л
!lstm_5/while/lstm_cell_10/BiasAddBiasAdd*lstm_5/while/lstm_cell_10/MatMul:product:0*lstm_5/while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_5/while/lstm_cell_10/BiasAddс
#lstm_5/while/lstm_cell_10/BiasAdd_1BiasAdd,lstm_5/while/lstm_cell_10/MatMul_1:product:0*lstm_5/while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/while/lstm_cell_10/BiasAdd_1с
#lstm_5/while/lstm_cell_10/BiasAdd_2BiasAdd,lstm_5/while/lstm_cell_10/MatMul_2:product:0*lstm_5/while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/while/lstm_cell_10/BiasAdd_2с
#lstm_5/while/lstm_cell_10/BiasAdd_3BiasAdd,lstm_5/while/lstm_cell_10/MatMul_3:product:0*lstm_5/while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/while/lstm_cell_10/BiasAdd_3Ц
lstm_5/while/lstm_cell_10/mul_4Mullstm_5_while_placeholder_2-lstm_5/while/lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_4Ц
lstm_5/while/lstm_cell_10/mul_5Mullstm_5_while_placeholder_2-lstm_5/while/lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_5Ц
lstm_5/while/lstm_cell_10/mul_6Mullstm_5_while_placeholder_2-lstm_5/while/lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_6Ц
lstm_5/while/lstm_cell_10/mul_7Mullstm_5_while_placeholder_2-lstm_5/while/lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_7Щ
(lstm_5/while/lstm_cell_10/ReadVariableOpReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(lstm_5/while/lstm_cell_10/ReadVariableOpЏ
-lstm_5/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-lstm_5/while/lstm_cell_10/strided_slice/stackГ
/lstm_5/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/lstm_5/while/lstm_cell_10/strided_slice/stack_1Г
/lstm_5/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/lstm_5/while/lstm_cell_10/strided_slice/stack_2
'lstm_5/while/lstm_cell_10/strided_sliceStridedSlice0lstm_5/while/lstm_cell_10/ReadVariableOp:value:06lstm_5/while/lstm_cell_10/strided_slice/stack:output:08lstm_5/while/lstm_cell_10/strided_slice/stack_1:output:08lstm_5/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2)
'lstm_5/while/lstm_cell_10/strided_sliceл
"lstm_5/while/lstm_cell_10/MatMul_4MatMul#lstm_5/while/lstm_cell_10/mul_4:z:00lstm_5/while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_4г
lstm_5/while/lstm_cell_10/addAddV2*lstm_5/while/lstm_cell_10/BiasAdd:output:0,lstm_5/while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/while/lstm_cell_10/addІ
!lstm_5/while/lstm_cell_10/SigmoidSigmoid!lstm_5/while/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!lstm_5/while/lstm_cell_10/SigmoidЭ
*lstm_5/while/lstm_cell_10/ReadVariableOp_1ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_1Г
/lstm_5/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/lstm_5/while/lstm_cell_10/strided_slice_1/stackЗ
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   23
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_1З
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_2Є
)lstm_5/while/lstm_cell_10/strided_slice_1StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_1:value:08lstm_5/while/lstm_cell_10/strided_slice_1/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_1/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_1н
"lstm_5/while/lstm_cell_10/MatMul_5MatMul#lstm_5/while/lstm_cell_10/mul_5:z:02lstm_5/while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_5й
lstm_5/while/lstm_cell_10/add_1AddV2,lstm_5/while/lstm_cell_10/BiasAdd_1:output:0,lstm_5/while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/add_1Ќ
#lstm_5/while/lstm_cell_10/Sigmoid_1Sigmoid#lstm_5/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/while/lstm_cell_10/Sigmoid_1Р
lstm_5/while/lstm_cell_10/mul_8Mul'lstm_5/while/lstm_cell_10/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_8Э
*lstm_5/while/lstm_cell_10/ReadVariableOp_2ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_2Г
/lstm_5/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   21
/lstm_5/while/lstm_cell_10/strided_slice_2/stackЗ
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   23
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_1З
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_2Є
)lstm_5/while/lstm_cell_10/strided_slice_2StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_2:value:08lstm_5/while/lstm_cell_10/strided_slice_2/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_2/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_2н
"lstm_5/while/lstm_cell_10/MatMul_6MatMul#lstm_5/while/lstm_cell_10/mul_6:z:02lstm_5/while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_6й
lstm_5/while/lstm_cell_10/add_2AddV2,lstm_5/while/lstm_cell_10/BiasAdd_2:output:0,lstm_5/while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/add_2
lstm_5/while/lstm_cell_10/TanhTanh#lstm_5/while/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
lstm_5/while/lstm_cell_10/TanhЦ
lstm_5/while/lstm_cell_10/mul_9Mul%lstm_5/while/lstm_cell_10/Sigmoid:y:0"lstm_5/while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/mul_9Ч
lstm_5/while/lstm_cell_10/add_3AddV2#lstm_5/while/lstm_cell_10/mul_8:z:0#lstm_5/while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/add_3Э
*lstm_5/while/lstm_cell_10/ReadVariableOp_3ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_3Г
/lstm_5/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   21
/lstm_5/while/lstm_cell_10/strided_slice_3/stackЗ
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_1З
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_2Є
)lstm_5/while/lstm_cell_10/strided_slice_3StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_3:value:08lstm_5/while/lstm_cell_10/strided_slice_3/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_3/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_3н
"lstm_5/while/lstm_cell_10/MatMul_7MatMul#lstm_5/while/lstm_cell_10/mul_7:z:02lstm_5/while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"lstm_5/while/lstm_cell_10/MatMul_7й
lstm_5/while/lstm_cell_10/add_4AddV2,lstm_5/while/lstm_cell_10/BiasAdd_3:output:0,lstm_5/while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2!
lstm_5/while/lstm_cell_10/add_4Ќ
#lstm_5/while/lstm_cell_10/Sigmoid_2Sigmoid#lstm_5/while/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#lstm_5/while/lstm_cell_10/Sigmoid_2Ѓ
 lstm_5/while/lstm_cell_10/Tanh_1Tanh#lstm_5/while/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_5/while/lstm_cell_10/Tanh_1Ь
 lstm_5/while/lstm_cell_10/mul_10Mul'lstm_5/while/lstm_cell_10/Sigmoid_2:y:0$lstm_5/while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 lstm_5/while/lstm_cell_10/mul_10
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder$lstm_5/while/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype023
1lstm_5/while/TensorArrayV2Write/TensorListSetItemj
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add/y
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/addn
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add_1/y
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/add_1
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/IdentityЃ
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_1
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_2И
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_3Ќ
lstm_5/while/Identity_4Identity$lstm_5/while/lstm_cell_10/mul_10:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/while/Identity_4Ћ
lstm_5/while/Identity_5Identity#lstm_5/while/lstm_cell_10/add_3:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
lstm_5/while/Identity_5"7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"h
1lstm_5_while_lstm_cell_10_readvariableop_resource3lstm_5_while_lstm_cell_10_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_10_split_1_readvariableop_resource;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_10_split_readvariableop_resource9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0"Ф
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ :џџџџџџџџџ : : :::2T
(lstm_5/while/lstm_cell_10/ReadVariableOp(lstm_5/while/lstm_cell_10/ReadVariableOp2X
*lstm_5/while/lstm_cell_10/ReadVariableOp_1*lstm_5/while/lstm_cell_10/ReadVariableOp_12X
*lstm_5/while/lstm_cell_10/ReadVariableOp_2*lstm_5/while/lstm_cell_10/ReadVariableOp_22X
*lstm_5/while/lstm_cell_10/ReadVariableOp_3*lstm_5/while/lstm_cell_10/ReadVariableOp_32`
.lstm_5/while/lstm_cell_10/split/ReadVariableOp.lstm_5/while/lstm_cell_10/split/ReadVariableOp2d
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: 
п
ё
-__inference_sequential_5_layer_call_fn_188572
lstm_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1885512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ&:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:џџџџџџџџџ&
&
_user_specified_namelstm_5_input
№	
н
D__inference_dense_15_layer_call_and_return_conditional_losses_190829

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
 
c
*__inference_dropout_5_layer_call_fn_190860

inputs
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1883852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ёј
в
while_body_187837
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resourceЂ!while/lstm_cell_10/ReadVariableOpЂ#while/lstm_cell_10/ReadVariableOp_1Ђ#while/lstm_cell_10/ReadVariableOp_2Ђ#while/lstm_cell_10/ReadVariableOp_3Ђ'while/lstm_cell_10/split/ReadVariableOpЂ)while/lstm_cell_10/split_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЈ
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"while/lstm_cell_10/ones_like/Constа
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/ones_like
 while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 while/lstm_cell_10/dropout/ConstЫ
while/lstm_cell_10/dropout/MulMul%while/lstm_cell_10/ones_like:output:0)while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2 
while/lstm_cell_10/dropout/Mul
 while/lstm_cell_10/dropout/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_10/dropout/Shape
7while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2єн29
7while/lstm_cell_10/dropout/random_uniform/RandomUniform
)while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2+
)while/lstm_cell_10/dropout/GreaterEqual/y
'while/lstm_cell_10/dropout/GreaterEqualGreaterEqual@while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2)
'while/lstm_cell_10/dropout/GreaterEqualИ
while/lstm_cell_10/dropout/CastCast+while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2!
while/lstm_cell_10/dropout/CastЦ
 while/lstm_cell_10/dropout/Mul_1Mul"while/lstm_cell_10/dropout/Mul:z:0#while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout/Mul_1
"while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_1/Constб
 while/lstm_cell_10/dropout_1/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout_1/Mul
"while/lstm_cell_10/dropout_1/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_1/Shape
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2т2;
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_1/GreaterEqual/y
)while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)while/lstm_cell_10/dropout_1/GreaterEqualО
!while/lstm_cell_10/dropout_1/CastCast-while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2#
!while/lstm_cell_10/dropout_1/CastЮ
"while/lstm_cell_10/dropout_1/Mul_1Mul$while/lstm_cell_10/dropout_1/Mul:z:0%while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2$
"while/lstm_cell_10/dropout_1/Mul_1
"while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_2/Constб
 while/lstm_cell_10/dropout_2/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout_2/Mul
"while/lstm_cell_10/dropout_2/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_2/Shape
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2х2;
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_2/GreaterEqual/y
)while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)while/lstm_cell_10/dropout_2/GreaterEqualО
!while/lstm_cell_10/dropout_2/CastCast-while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2#
!while/lstm_cell_10/dropout_2/CastЮ
"while/lstm_cell_10/dropout_2/Mul_1Mul$while/lstm_cell_10/dropout_2/Mul:z:0%while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2$
"while/lstm_cell_10/dropout_2/Mul_1
"while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_3/Constб
 while/lstm_cell_10/dropout_3/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2"
 while/lstm_cell_10/dropout_3/Mul
"while/lstm_cell_10/dropout_3/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_3/Shape
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&*
dtype0*
seedБџх)*
seed2ин2;
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_3/GreaterEqual/y
)while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ&2+
)while/lstm_cell_10/dropout_3/GreaterEqualО
!while/lstm_cell_10/dropout_3/CastCast-while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ&2#
!while/lstm_cell_10/dropout_3/CastЮ
"while/lstm_cell_10/dropout_3/Mul_1Mul$while/lstm_cell_10/dropout_3/Mul:z:0%while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ&2$
"while/lstm_cell_10/dropout_3/Mul_1
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$while/lstm_cell_10/ones_like_1/Constи
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
while/lstm_cell_10/ones_like_1
"while/lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_4/Constг
 while/lstm_cell_10/dropout_4/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_4/Mul
"while/lstm_cell_10/dropout_4/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_4/Shape
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2зО2;
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_4/GreaterEqual/y
)while/lstm_cell_10/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_4/GreaterEqualО
!while/lstm_cell_10/dropout_4/CastCast-while/lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_4/CastЮ
"while/lstm_cell_10/dropout_4/Mul_1Mul$while/lstm_cell_10/dropout_4/Mul:z:0%while/lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_4/Mul_1
"while/lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_5/Constг
 while/lstm_cell_10/dropout_5/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_5/Mul
"while/lstm_cell_10/dropout_5/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_5/Shape
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2єзЎ2;
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_5/GreaterEqual/y
)while/lstm_cell_10/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_5/GreaterEqualО
!while/lstm_cell_10/dropout_5/CastCast-while/lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_5/CastЮ
"while/lstm_cell_10/dropout_5/Mul_1Mul$while/lstm_cell_10/dropout_5/Mul:z:0%while/lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_5/Mul_1
"while/lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_6/Constг
 while/lstm_cell_10/dropout_6/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_6/Mul
"while/lstm_cell_10/dropout_6/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_6/Shape
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2и2;
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_6/GreaterEqual/y
)while/lstm_cell_10/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_6/GreaterEqualО
!while/lstm_cell_10/dropout_6/CastCast-while/lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_6/CastЮ
"while/lstm_cell_10/dropout_6/Mul_1Mul$while/lstm_cell_10/dropout_6/Mul:z:0%while/lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_6/Mul_1
"while/lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"while/lstm_cell_10/dropout_7/Constг
 while/lstm_cell_10/dropout_7/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 while/lstm_cell_10/dropout_7/Mul
"while/lstm_cell_10/dropout_7/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_7/Shape
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0*
seedБџх)*
seed2ОПs2;
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniform
+while/lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2-
+while/lstm_cell_10/dropout_7/GreaterEqual/y
)while/lstm_cell_10/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2+
)while/lstm_cell_10/dropout_7/GreaterEqualО
!while/lstm_cell_10/dropout_7/CastCast-while/lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ 2#
!while/lstm_cell_10/dropout_7/CastЮ
"while/lstm_cell_10/dropout_7/Mul_1Mul$while/lstm_cell_10/dropout_7/Mul:z:0%while/lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"while/lstm_cell_10/dropout_7/Mul_1С
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mulЧ
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_1Ч
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_2Ч
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЦ
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&*
dtype02)
'while/lstm_cell_10/split/ReadVariableOpѓ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/splitБ
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMulЗ
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_1З
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_2З
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dimШ
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOpы
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1П
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAddХ
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_1Х
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_2Х
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/BiasAdd_3Њ
while/lstm_cell_10/mul_4Mulwhile_placeholder_2&while/lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_4Њ
while/lstm_cell_10/mul_5Mulwhile_placeholder_2&while/lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_5Њ
while/lstm_cell_10/mul_6Mulwhile_placeholder_2&while/lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_6Њ
while/lstm_cell_10/mul_7Mulwhile_placeholder_2&while/lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_7Д
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02#
!while/lstm_cell_10/ReadVariableOpЁ
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stackЅ
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1Ѕ
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2ю
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_sliceП
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_4З
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/SigmoidИ
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_1Ѕ
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stackЉ
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1Љ
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2њ
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1С
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_5Н
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_1Є
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_8И
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_2Ѕ
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stackЉ
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1Љ
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2њ
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2С
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_6Н
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_2
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/TanhЊ
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_9Ћ
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_3И
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 *
dtype02%
#while/lstm_cell_10/ReadVariableOp_3Ѕ
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stackЉ
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1Љ
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2њ
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3С
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/MatMul_7Н
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/add_4
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/Tanh_1А
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/lstm_cell_10/mul_10с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ъ
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityн
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ь
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2љ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3э
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_4ь
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ :џџџџџџџџџ : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: "БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Й
serving_defaultЅ
I
lstm_5_input9
serving_default_lstm_5_input:0џџџџџџџџџ&<
dense_170
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:рђ
­5
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
*u&call_and_return_all_conditional_losses
v_default_save_signature
w__call__"Й2
_tf_keras_sequential2{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_5_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "recurrent_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 38]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 38]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_5_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "recurrent_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Њ
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
*x&call_and_return_all_conditional_losses
y__call__"
_tf_keras_rnn_layerу{"class_name": "LSTM", "name": "lstm_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "recurrent_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 38]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 38]}}
ѕ

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*z&call_and_return_all_conditional_losses
{__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
х
	variables
regularization_losses
trainable_variables
	keras_api
*|&call_and_return_all_conditional_losses
}__call__"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
ѕ

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
*~&call_and_return_all_conditional_losses
__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
і

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+&call_and_return_all_conditional_losses
__call__"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
ѕ
(iter

)beta_1

*beta_2
	+decay
,learning_ratemcmdmemf"mg#mh-mi.mj/mkvlvmvnvo"vp#vq-vr.vs/vt"
	optimizer
_
-0
.1
/2
3
4
5
6
"7
#8"
trackable_list_wrapper
 "
trackable_list_wrapper
_
-0
.1
/2
3
4
5
6
"7
#8"
trackable_list_wrapper
Ъ
0metrics
	variables

1layers
2layer_regularization_losses
3non_trainable_variables
4layer_metrics
regularization_losses
	trainable_variables
w__call__
v_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
	

-kernel
.recurrent_kernel
/bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
+&call_and_return_all_conditional_losses
__call__"с
_tf_keras_layerЧ{"class_name": "LSTMCell", "name": "lstm_cell_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "recurrent_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}}
 "
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
Й
9metrics
	variables

:layers
;layer_regularization_losses

<states
=non_trainable_variables
>layer_metrics
regularization_losses
trainable_variables
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_15/kernel
: 2dense_15/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
?metrics
	variables

@layers
Alayer_regularization_losses
Bnon_trainable_variables
Clayer_metrics
regularization_losses
trainable_variables
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dmetrics
	variables

Elayers
Flayer_regularization_losses
Gnon_trainable_variables
Hlayer_metrics
regularization_losses
trainable_variables
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_16/kernel
: 2dense_16/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Imetrics
	variables

Jlayers
Klayer_regularization_losses
Lnon_trainable_variables
Mlayer_metrics
regularization_losses
 trainable_variables
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_17/kernel
:2dense_17/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
А
Nmetrics
$	variables

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rlayer_metrics
%regularization_losses
&trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2iter
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
-:+	&2lstm_5/lstm_cell_10/kernel
7:5	 2$lstm_5/lstm_cell_10/recurrent_kernel
':%2lstm_5/lstm_cell_10/bias
.
S0
T1"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
-0
.1
/2"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
А
Umetrics
5	variables

Vlayers
Wlayer_regularization_losses
Xnon_trainable_variables
Ylayer_metrics
6regularization_losses
7trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
Л
	Ztotal
	[count
\	variables
]	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
џ
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api"И
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
Z0
[1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
^0
_1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
!:  2dense_15/kernel/m
: 2dense_15/bias/m
!:  2dense_16/kernel/m
: 2dense_16/bias/m
!: 2dense_17/kernel/m
:2dense_17/bias/m
-:+	&2lstm_5/lstm_cell_10/kernel/m
7:5	 2&lstm_5/lstm_cell_10/recurrent_kernel/m
':%2lstm_5/lstm_cell_10/bias/m
!:  2dense_15/kernel/v
: 2dense_15/bias/v
!:  2dense_16/kernel/v
: 2dense_16/bias/v
!: 2dense_17/kernel/v
:2dense_17/bias/v
-:+	&2lstm_5/lstm_cell_10/kernel/v
7:5	 2&lstm_5/lstm_cell_10/recurrent_kernel/v
':%2lstm_5/lstm_cell_10/bias/v
ю2ы
H__inference_sequential_5_layer_call_and_return_conditional_losses_189392
H__inference_sequential_5_layer_call_and_return_conditional_losses_189103
H__inference_sequential_5_layer_call_and_return_conditional_losses_188509
H__inference_sequential_5_layer_call_and_return_conditional_losses_188470Р
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
ш2х
!__inference__wrapped_model_186803П
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ */Ђ,
*'
lstm_5_inputџџџџџџџџџ&
2џ
-__inference_sequential_5_layer_call_fn_188634
-__inference_sequential_5_layer_call_fn_188572
-__inference_sequential_5_layer_call_fn_189438
-__inference_sequential_5_layer_call_fn_189415Р
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
ы2ш
B__inference_lstm_5_layer_call_and_return_conditional_losses_190529
B__inference_lstm_5_layer_call_and_return_conditional_losses_189845
B__inference_lstm_5_layer_call_and_return_conditional_losses_190796
B__inference_lstm_5_layer_call_and_return_conditional_losses_190112е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
џ2ќ
'__inference_lstm_5_layer_call_fn_190818
'__inference_lstm_5_layer_call_fn_190123
'__inference_lstm_5_layer_call_fn_190807
'__inference_lstm_5_layer_call_fn_190134е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
D__inference_dense_15_layer_call_and_return_conditional_losses_190829Ђ
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
г2а
)__inference_dense_15_layer_call_fn_190838Ђ
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
Ш2Х
E__inference_dropout_5_layer_call_and_return_conditional_losses_190850
E__inference_dropout_5_layer_call_and_return_conditional_losses_190855Д
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
2
*__inference_dropout_5_layer_call_fn_190865
*__inference_dropout_5_layer_call_fn_190860Д
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
ю2ы
D__inference_dense_16_layer_call_and_return_conditional_losses_190876Ђ
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
г2а
)__inference_dense_16_layer_call_fn_190885Ђ
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
ю2ы
D__inference_dense_17_layer_call_and_return_conditional_losses_190896Ђ
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
г2а
)__inference_dense_17_layer_call_fn_190905Ђ
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
аBЭ
$__inference_signature_wrapper_188679lstm_5_input"
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
 
и2е
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_191173
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_191077О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

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
Ђ2
-__inference_lstm_cell_10_layer_call_fn_191190
-__inference_lstm_cell_10_layer_call_fn_191207О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

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
Г2А
__inference_loss_fn_0_191218
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_1_191229
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ  
!__inference__wrapped_model_186803{	-/."#9Ђ6
/Ђ,
*'
lstm_5_inputџџџџџџџџџ&
Њ "3Њ0
.
dense_17"
dense_17џџџџџџџџџЄ
D__inference_dense_15_layer_call_and_return_conditional_losses_190829\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 |
)__inference_dense_15_layer_call_fn_190838O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Є
D__inference_dense_16_layer_call_and_return_conditional_losses_190876\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 |
)__inference_dense_16_layer_call_fn_190885O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Є
D__inference_dense_17_layer_call_and_return_conditional_losses_190896\"#/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_dense_17_layer_call_fn_190905O"#/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЅ
E__inference_dropout_5_layer_call_and_return_conditional_losses_190850\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ "%Ђ"

0џџџџџџџџџ 
 Ѕ
E__inference_dropout_5_layer_call_and_return_conditional_losses_190855\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ "%Ђ"

0џџџџџџџџџ 
 }
*__inference_dropout_5_layer_call_fn_190860O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ "џџџџџџџџџ }
*__inference_dropout_5_layer_call_fn_190865O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ "џџџџџџџџџ ;
__inference_loss_fn_0_191218-Ђ

Ђ 
Њ " ;
__inference_loss_fn_1_191229/Ђ

Ђ 
Њ " Г
B__inference_lstm_5_layer_call_and_return_conditional_losses_189845m-/.?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ&

 
p

 
Њ "%Ђ"

0џџџџџџџџџ 
 Г
B__inference_lstm_5_layer_call_and_return_conditional_losses_190112m-/.?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ&

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ 
 У
B__inference_lstm_5_layer_call_and_return_conditional_losses_190529}-/.OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ&

 
p

 
Њ "%Ђ"

0џџџџџџџџџ 
 У
B__inference_lstm_5_layer_call_and_return_conditional_losses_190796}-/.OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ&

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ 
 
'__inference_lstm_5_layer_call_fn_190123`-/.?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ&

 
p

 
Њ "џџџџџџџџџ 
'__inference_lstm_5_layer_call_fn_190134`-/.?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ&

 
p 

 
Њ "џџџџџџџџџ 
'__inference_lstm_5_layer_call_fn_190807p-/.OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ&

 
p

 
Њ "џџџџџџџџџ 
'__inference_lstm_5_layer_call_fn_190818p-/.OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ&

 
p 

 
Њ "џџџџџџџџџ Ъ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_191077§-/.Ђ}
vЂs
 
inputsџџџџџџџџџ&
KЂH
"
states/0џџџџџџџџџ 
"
states/1џџџџџџџџџ 
p
Њ "sЂp
iЂf

0/0џџџџџџџџџ 
EB

0/1/0џџџџџџџџџ 

0/1/1џџџџџџџџџ 
 Ъ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_191173§-/.Ђ}
vЂs
 
inputsџџџџџџџџџ&
KЂH
"
states/0џџџџџџџџџ 
"
states/1џџџџџџџџџ 
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџ 
EB

0/1/0џџџџџџџџџ 

0/1/1џџџџџџџџџ 
 
-__inference_lstm_cell_10_layer_call_fn_191190э-/.Ђ}
vЂs
 
inputsџџџџџџџџџ&
KЂH
"
states/0џџџџџџџџџ 
"
states/1џџџџџџџџџ 
p
Њ "cЂ`

0џџџџџџџџџ 
A>

1/0џџџџџџџџџ 

1/1џџџџџџџџџ 
-__inference_lstm_cell_10_layer_call_fn_191207э-/.Ђ}
vЂs
 
inputsџџџџџџџџџ&
KЂH
"
states/0џџџџџџџџџ 
"
states/1џџџџџџџџџ 
p 
Њ "cЂ`

0џџџџџџџџџ 
A>

1/0џџџџџџџџџ 

1/1џџџџџџџџџ С
H__inference_sequential_5_layer_call_and_return_conditional_losses_188470u	-/."#AЂ>
7Ђ4
*'
lstm_5_inputџџџџџџџџџ&
p

 
Њ "%Ђ"

0џџџџџџџџџ
 С
H__inference_sequential_5_layer_call_and_return_conditional_losses_188509u	-/."#AЂ>
7Ђ4
*'
lstm_5_inputџџџџџџџџџ&
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Л
H__inference_sequential_5_layer_call_and_return_conditional_losses_189103o	-/."#;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ&
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Л
H__inference_sequential_5_layer_call_and_return_conditional_losses_189392o	-/."#;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ&
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
-__inference_sequential_5_layer_call_fn_188572h	-/."#AЂ>
7Ђ4
*'
lstm_5_inputџџџџџџџџџ&
p

 
Њ "џџџџџџџџџ
-__inference_sequential_5_layer_call_fn_188634h	-/."#AЂ>
7Ђ4
*'
lstm_5_inputџџџџџџџџџ&
p 

 
Њ "џџџџџџџџџ
-__inference_sequential_5_layer_call_fn_189415b	-/."#;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ&
p

 
Њ "џџџџџџџџџ
-__inference_sequential_5_layer_call_fn_189438b	-/."#;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ&
p 

 
Њ "џџџџџџџџџД
$__inference_signature_wrapper_188679	-/."#IЂF
Ђ 
?Њ<
:
lstm_5_input*'
lstm_5_inputџџџџџџџџџ&"3Њ0
.
dense_17"
dense_17џџџџџџџџџ