
function InstallCpuOps(builder) {
    function GenerateMLOperandFromNumber(val, type = 'int32') {
       if((!Number.isInteger(val) || type == 'float32')&& Number.isFinite(val))
       {
            // float case
            let array_buffer_view = new Float32Array(new ArrayBuffer(4));
            array_buffer_view[0] = val;
            return builder.constant({type: 'float32', dataType: 'float32', dimensions: [1]}, array_buffer_view);
       } else if (Number.isInteger(val))
       {
            let array_buffer_view = null;
            if (type == 'int32')
            {
                // Integer case
                array_buffer_view = new Int32Array(new ArrayBuffer(4));
                array_buffer_view[0] = val;
            }
            else if (type = 'int8')
            {
                array_buffer_view = new Int8Array(new ArrayBuffer(1));
                array_buffer_view[0] = val;
            }
            return builder.constant({type: type, dataType: type, dimensions: [1]}, array_buffer_view);
       }
       throw("GenerateMLOperandFromNumber non float is not yet implemented.");
    }
    function GenerateMLOperandFromArray(val, type = 'int32') {
        if (Number.isInteger(val[0]))
        {
             // Integer case
             let array_buffer_view;
             switch(type)
             {
                case 'int32':
                    array_buffer_view = new Int32Array(val);
                    break;
                case 'uint8':
                    array_buffer_view = new Uint8Array(val);
                    break;
                case 'float32':
                    array_buffer_view = new Float32Array(val);
                    break;
             }
             return builder.constant({type: type, dataType: type, dimensions: [val.length]}, array_buffer_view);
        }
        throw("GenerateMLOperandFromArray is not yet implemented.");
    }
    function mul(original) {
        return function (...args) {
            if (typeof args[0] === 'number' && typeof args[1] === 'number')
            {
                return args[0] * args[1];
            }
            if (typeof args[1] === 'number')
            {
                args[1] = GenerateMLOperandFromNumber(args[1], args[0].dataType());
            }
            return original.apply(this, args);
        };
    }
    function div(original) {
        return function (...args) {
            if (typeof args[0] === 'number' && typeof args[1] === 'number')
            {
                return args[0] / args[1];
            }
            if (typeof args[0] === 'number')
            {
                args[0] = GenerateMLOperandFromNumber(args[0], args[1].dataType());
            }
            return original.apply(this, args);
        };
    }
    function add(original) {
        return function (...args) {
            if (typeof args[0] === 'number' && typeof args[1] === 'number')
            {
                return args[0] + args[1];
            }
            if (typeof args[1] === 'number')
            {
                args[1] = GenerateMLOperandFromNumber(args[1]);
            }
            return original.apply(this, args);
        };
    }
    function sub(original) {
        return function (...args) {
            if (typeof args[0] === 'number' && typeof args[1] === 'number')
            {
                return args[0] - args[1];
            }
            if (typeof args[0] === 'number')
            {
                args[0] = GenerateMLOperandFromNumber(args[0], args[1].dataType());
            }
            return original.apply(this, args);
        };
    }
    function pow(original) {
        return function (...args) {
            if (typeof args[1] === 'number')
            {
                args[1] = GenerateMLOperandFromNumber(args[1], args[0].dataType());
            }
            return original.apply(this, args);
        };
    }
    function sqrt(original) {
        return function (...args) {
            if (args[0] instanceof Array)
            {
                args[0] = GenerateMLOperandFromArray(args[0], 'float32');
            }
            return original.apply(this, args);
        };
    }
    function reshape(original) {
        return function (...args) {
            if (typeof args[0] === 'number')
            {
                args[0] = GenerateMLOperandFromNumber(args[0], 'float32');
            } 
            else if (args[0] instanceof Array)
            {
                if (typeof args[1] === 'number')
                {
                    return args[0];
                }
                else
                {
                    throw("CPU OP reshape is not fully implemented.");
                }
            }
            return original.apply(this, args);
        };
    }
    function equal(original) {
        return function (...args) {
            if (args[0] instanceof Array)
            {
                if (typeof args[0][1] === 'number')
                {
                    args[0] = GenerateMLOperandFromArray(args[0], args[1].dataType());
                }
                else
                {
                    throw("CPU OP equal is not fully implemented.");
                }
            }
            return original.apply(this, args);
        };
    }
    function where(original) {
        return function (...args) {
            if (args[0] instanceof Array)
            {
                if (typeof args[0][1] === 'number')
                {
                    args[0] = GenerateMLOperandFromArray(args[0], 'uint8');
                }
                else
                {
                    throw("CPU OP where is not fully implemented.");
                }
            }
            if (args[2] instanceof Array)
            {
                args[2] = GenerateMLOperandFromArray(args[2], args[1].dataType());
            }
            if (typeof args[1] === 'number')
            {
                args[1] = GenerateMLOperandFromNumber(args[1], args[2].dataType());
            }
            return original.apply(this, args);
        };
    }
    function range(original) {
        return function (...args) {
            let array = [];
            // Inputs are start, limit, delta
            while(true) {
                let number = args[0] + (array.length * args[2]);
                if (number >= args[1])
                    break;
                array.push(number);
            }
            return GenerateMLOperandFromArray(array);
        };
    }
    function cast(original) {
        return function (...args) {
            if (typeof args[0] === 'number' && args[1] == "int64")
            {
                return args[0];
            }
            if (typeof args[0] === 'number' && args[1] == "float32")
            {
                return args[0];
            }
            if (args[0] instanceof Array && args[1] == "int64" && args[0].length == 1)
            {
                return BigInt(args[0][0]);
            }
            if (args[0] instanceof Array && args[1] == "float32")
            {
                return args[0];
            }
            return original.apply(this, args);
        };
    }
    function gather(original) {
        return function (...args) {
            if (typeof args[1] === 'number' && args[2].axis === 0)
            {
                // Super simple case, where indices must just index
                // into input.
                return args[0][args[1]];
            }
            return original.apply(this, args);
        };
    }
    function unsqueeze(original) {
        return function (...args) {
            if (args[1] == 0 && typeof args[0] === 'number')
            {
                return [args[0]];
            }
            else if (args[0] instanceof MLOperand)
            {
                let new_shape = args[0].shape();
                new_shape.splice(args[1], 0, 1);
                return this.reshape(args[0], new_shape);
            }
            throw("CPU OP Unsqueeze is not fully implemented.");
        };
    }
    function concat(original) {
        return function (...args) {
            if (args[0][0] instanceof NullTensor)
            {
                return args[0][1];
            }
            if ((Array.isArray(args[0][0]) || typeof args[0][0] === 'number') && args[1] === 0)
            {
                // args[1] is the axis, args[0] is the array of tensors we need to concat
                // Just calling flat will create a concatenated tensor with those numbers.
                return args[0].flat(1);
            }
            if (args[1] < 0)
            {
                args[1] = args[0][0].shape().length + args[1];
            }
            return original.apply(this, args);
        };
    }
    function slice(original) {
        return function (...args) {
            // value, start, end, axis
            if (Array.isArray(args[0]) && args[1] == -1 && (args.length == 3 || args[3] == 0) && args[2] > args[0].length)
            {
                // Super simple case, where indices must just index
                // into input.
                let length = args[0].length;
                let value = args[0][length-1];
                return [value];
            }
            if (args[4] == 1 && typeof args[3] === 'number')
            {
                // WebNN IDL looks like
                //  [RaisesException] MLOperand slice(MLOperand input, 
                //     sequence<[EnforceRange] unsigned long> starts, 
                //     sequence<[EnforceRange] unsigned long> sizes);
                // For now we know how to implement when the step is 1, and axis is 0.
                if (typeof args[1] === 'number')
                {
                    // Wrap it in an array.
                    args[1] = [args[1]];
                }
                if (typeof args[2] === 'number')
                {
                    // Wrap it in an array.
                    args[2] = [args[2]];
                }
                if (!Array.isArray(args[1]) || !Array.isArray(args[2]) || args[1].length != args[2].length)
                {
                    throw("Slice is not fully implemented.");        
                }
                // Convert end to size notation.
                for (let i =0; i < args[2].length; i++)
                {
                    args[2][i] = args[2][i] - args[1][i];
                }
                // Fill in the other axis
                let shape = args[0].shape();
                let new_starts = [];
                let new_size = [];
                for (let i = 0; i < shape.length; i++)
                {
                    new_starts.push(0);
                    new_size.push(shape[i]);
                }
                new_starts[args[3]] = args[1][0];
                new_size[args[3]] = args[2][0];
                for (let i = 0; i < shape.length; i++)
                {
                    if (new_starts[i] > shape[i])
                    {
                        new_starts[i] = shape[i]
                    }
                    if (new_size[i] > shape[i])
                    {
                        new_size[i] = shape[i] - new_starts[i];
                    }
                }
                return original.apply(this,[args[0], new_starts, new_size]);
            }
            throw("CPU OP slice is not fully implemented.");
        };
    }
    function squeeze(original){
        return function (...args) {
            if (Array.isArray(args[0]) && args[1] == 0)
            {
                if (args[0].length == 1) {
                    return args[0][0];
                } else {
                    // The dimension is non zero return the array as is. 
                    return args[0];
                }
            }
            throw("CPU OP squeeze is not fully implemented.");
        };
    }
    function QuantizeLinear(...args) 
    {
        // y = saturate (round (x / y_scale) + y_zero_point)
        let div_result = this.div(args[0], args[1]);
        let round_result = this.add(args[0], this.constant_dql_pt5);
        let add_y_zp_result = this.add(round_result, args[2]);
        let saturate_result = this.clamp(add_y_zp_result, {minValue:0, maxValue:255});
        return this.cast(saturate_result, "uint8");
    }
    function generateConstantOfShape(type, value, shape)
    {
        let size = 1;
        for (const dim of shape) {
            size *= dim;
        }
        if (size == 0)
        {
            return new NullTensor(shape);
        }
        let data = null;
        let array_buffer_view;
        switch(type)
        {
           case 'int64':
               array_buffer_view = new BigInt64Array(size);
               value = BigInt(value);
               break;
           case 'int32':
               array_buffer_view = new Int32Array(size);
               break;
           case 'uint8':
               array_buffer_view = new Uint8Array(size);
               break;
           case 'float32':
               array_buffer_view = new Float32Array(size);
               break;
        }
        array_buffer_view.fill(value);
        return builder.constant({type: type, dataType: type, dimensions: shape}, array_buffer_view);
    }

    function cpu_generateConstantOfShape(type, value, shape)
    {
        if (shape.length != 1)
        {
            throw("CPU OP cpu_generateConstantOfShape is not fully implemented.");
        }
        let result = new Array(shape[0]);
        result = result.fill(value,0,shape[0]);
        result.type = type;
        return result;
    }
    function cpu_concat(sequence, axis)
    {
        if (axis != 0)
        {
            throw("CPU OP concat is not fully implemented.");
        }
        let result=[];
        for (const input of sequence) 
        {
            if (!Array.isArray(input) && !Number.isFinite(input))
            {
                throw("CPU OP concat is not fully implemented.");
            }
            result = result.concat(input);
        }
        return result;
    }
    function cpu_gather(...args)
    {
        if (typeof args[1] === 'number' && args[2].axis === 0)
        {
            // Super simple case, where indices must just index
            // into input.
            return args[0][args[1]];
        }
        throw("CPU OP gather is not fully implemented.");
    }
    function cpu_add(...args)
    {
        if (typeof args[0] === 'number' && typeof args[1] === 'number')
        {
            return args[0] + args[1];
        }
        throw("CPU OP add is not fully implemented.");
    }
    function cpu_sub(...args)
    {
        if (typeof args[0] === 'number' && typeof args[1] === 'number')
        {
            return args[0] - args[1];
        }
        throw("CPU OP sub is not fully implemented.");
    }
    function cpu_unsqueeze(...args)
    {
        if (args[1] == 0 && typeof args[0] === 'number')
        {
            return [args[0]];
        }
        throw("CPU OP unsqueeze is not fully implemented.");
    }
    function cpu_equal(...args)
    {
        output = [];
        for (let i=0; i < args[0].length || i < args[1].length; i++)
        {
            let lhs = i < args[0].length ? args[0][i] : args[0][args[0].length - 1];
            let rhs = i < args[1].length ? args[1][i] : args[1][args[1].length - 1];
            output.push(lhs == rhs ? 1 : 0);
        }
        return output;
    }
    function cpu_reshape(...args)
    {
        if (args[1] == -1)
            return args[0];
        if (Array.isArray(args[1]) && args[1].length == 1 && args[1][0] == 1 && typeof args[0] === 'number')
        {
            return [args[0]];
        }
        throw("CPU OP reshape is not fully implemented.");
    }
    function cpu_where(...args)
    {
        output = [];
        for (let i=0; i < args[0].length; i++)
        {
            let cond = i < args[0].length ? args[0][i] : args[0][args[0].length - 1];
            let tval = i < args[1].length ? args[1][i] : args[1][args[1].length - 1];
            let fval = i < args[2].length ? args[2][i] : args[2][args[2].length - 1];
            output.push(cond?tval:fval);
        }
        return output;
    }
    function cpu_mul(...args)
    {
        output = [];
        if (typeof args[1] == 'number') {
            args[1] = [args[1]];
        }
        for (let i=0; i < args[0].length || i < args[1].length; i++)
        {
            let lhs = i < args[0].length ? args[0][i] : args[0][args[0].length - 1];
            let rhs = i < args[1].length ? args[1][i] : args[1][args[1].length - 1];
            output.push(lhs * rhs);
        }
        return output;
    }

    builder.mul = mul(builder.mul);
    builder.add = add(builder.add);
    builder.sub = sub(builder.sub);
    builder.pow = pow(builder.pow);
    builder.sqrt = sqrt(builder.sqrt);
    builder.div = div(builder.div);
    builder.cast = cast(builder.cast);
    builder.reshape = reshape(builder.reshape);
    builder.gather = gather(builder.gather);
    builder.unsqueeze = unsqueeze(builder.unsqueeze);
    builder.concat = concat(builder.concat);
    builder.slice = slice(builder.slice);
    builder.squeeze = squeeze(builder.squeeze);
    builder.equal = equal(builder.equal);
    builder.range = range(builder.range);
    builder.where = where(builder.where);
    builder.QuantizeLinear = QuantizeLinear;
    builder.generateConstantOfShape = generateConstantOfShape;

    builder.cpu_gather = cpu_gather;
    builder.cpu_add = cpu_add;
    builder.cpu_sub = cpu_sub;
    builder.cpu_unsqueeze = cpu_unsqueeze;
    builder.cpu_equal = cpu_equal;
    builder.cpu_reshape = cpu_reshape;
    builder.cpu_where = cpu_where;
    builder.cpu_mul = cpu_mul;
    builder.cpu_concat = cpu_concat;
    builder.cpu_generateConstantOfShape = cpu_generateConstantOfShape;

    builder.constant_dql_255 = GenerateMLOperandFromNumber(255.0, 'float32');
    builder.constant_dql_pt5 = GenerateMLOperandFromNumber(0.5, 'float32');
    Array.prototype.shape = function() { return [this.length]; }
}

class NullTensor
{
    constructor(dims)   
    {
        this.dims = dims;
    }
    shape()
    {
        return this.dims;
    }
}