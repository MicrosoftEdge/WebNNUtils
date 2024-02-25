
function InstallCpuOps(builder) {
    function mul(original) {
        return function (...args) {
            if (typeof args[0] === 'number' && typeof args[1] === 'number')
            {
                return args[0] * args[1];
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
            return original.apply(this, args);
        };
    }
    function cast(original) {
        return function (...args) {
            if (typeof args[0] === 'number' && args[1] == "int64")
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
            if (args[1] == 0)
            {
                return [args[0]];
            }
            throw("CPU OP Unsqueeze is not fully implemented.");
        };
    }
    function concat(original) {
        return function (...args) {
            if ((Array.isArray(args[0][0]) || typeof args[0][0] === 'number') && args[1] === 0)
            {
                // Super simple case, where indices must just index
                // into input.
                return args[0].flat(1);
            }
            return original.apply(this, args);
        };
    }

    builder.mul = mul(builder.mul);
    builder.div = div(builder.div);
    builder.cast = cast(builder.cast);
    builder.gather = gather(builder.gather);
    builder.unsqueeze = unsqueeze(builder.unsqueeze);
    builder.concat = concat(builder.concat);
}