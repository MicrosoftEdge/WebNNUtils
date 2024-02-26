// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import { DataType } from './Wasm-Common.js';
import { ShapeUtil } from './Util.js';
/**
 * constant value for a workgroup size.
 *
 * We definitely can do further optimization in future, but for now we use 64.
 *
 * rule of thumb: Use [a workgroup size of] 64 unless you know what GPU you are targeting or that your workload
 *                needs something different.
 *
 * from: https://surma.dev/things/webgpu/
 **/
export const WORKGROUP_SIZE = 64;
const getWgslMappedType = (type, components) => {
    if (components === 3) {
        throw new Error('vec3 has same alignment as vec4, use vec4 instead');
    }
    // return type is [ storage type, runtime type ] or a single string for both
    switch (type) {
        case DataType.float16:
            return components > 1 ? `vec${components}<f16>` : 'f16';
        case DataType.float:
            return components > 1 ? `vec${components}<f32>` : 'f32';
        case DataType.int32:
            return components > 1 ? `vec${components}<i32>` : 'i32';
        case DataType.uint32:
            return components > 1 ? `vec${components}<u32>` : 'u32';
        case DataType.int64:
            if (components > 1) {
                throw new Error('currently not supported vecX of uint64 yet');
            }
            return ['vec2<u32>', 'i32'];
        case DataType.uint64:
            if (components > 1) {
                throw new Error('currently not supported vecX of uint64 yet');
            }
            return ['vec2<u32>', 'u32'];
        case DataType.bool:
            if (components !== 4) {
                throw new Error('bool must be vec4');
            }
            return ['u32', 'vec4<bool>'];
        default:
            throw new Error(`Unknown data type: ${type}`);
    }
};
export const tensorTypeToWsglStorageType = (type, components = 1) => {
    const mappedType = getWgslMappedType(type, components);
    return typeof mappedType === 'string' ? mappedType : mappedType[0];
};
export const tensorTypeToWsglValueType = (type, components = 1) => {
    const mappedType = getWgslMappedType(type, components);
    return typeof mappedType === 'string' ? mappedType : mappedType[1];
};
export const createTensorShapeVariables = (...dims) => {
    const programUniforms = [];
    dims.forEach(dim => {
        if (dim.length !== 0) {
            programUniforms.push({ type: DataType.uint32, data: dim }, { type: DataType.uint32, data: ShapeUtil.computeStrides(dim) });
        }
    });
    return programUniforms;
};
/**
 * A helper function to get maximum vector size for specified data length
 * @param size
 */
export const getMaxComponents = (size) => {
    // we cannot use vec3 type since it has alignment of 16 bytes
    if (size % 4 === 0) {
        return 4;
    }
    else if (size % 2 === 0) {
        return 2;
    }
    return 1;
};
/**
 * A helper function that initializes variable as a scalar or vector. e.g. f32(0) or vec4f(0,0,0,0)
 * @param dataType
 * @param components
 * @param value
 */
export const fillVector = (dataType = 'f32', components, value = '0') => {
    if (!components || components === 1) {
        return `${dataType}(${value})`;
    }
    return `vec${components}<${dataType}>(${value})`;
};
/**
 * A helper function that casts value or vector to f32
 * @param dataType
 * @param components
 * @param value
 */
export const castToF32 = (dataType, components, value) => {
    if (dataType === 'f32') {
        return value;
    }
    if (components === 1) {
        return `f32(${value})`;
    }
    return `vec${components}f(${value})`;
};
/**
 * A helper function that returns scalar or sums all components of a vector
 * @param name
 * @param components
 */
export const sumVector = (name, components) => {
    if (components === 4) {
        return `(${name}.x + ${name}.y + ${name}.z + ${name}.w)`;
    }
    else if (components === 2) {
        return `(${name}.x + ${name}.y)`;
    }
    else if (components === 3) {
        return `(${name}.x + ${name}.y + ${name}.z)`;
    }
    return name;
};
/**
 * A helper function that returns variable element at index.
 * @param name - the name of variable.
 * @param index - the index of variable element.
 * @param length - the length of variable.
 * @param type - the type of variable, optional.
 */
export const getElementAt = (name, index, length, type) => {
    if (name.startsWith('uniforms.') && length > 4) {
        if (typeof (index) === 'string') {
            if (type === 'f16') {
                return `${name}[(${index}) / 8][(${index}) % 8 / 4][(${index}) % 8 % 4]`;
            }
            else {
                return `${name}[(${index}) / 4][(${index}) % 4]`;
            }
        }
        else {
            if (type === 'f16') {
                return `${name}[${Math.floor(index / 8)}][${Math.floor(index % 8 / 4)}][${index % 8 % 4}]`;
            }
            else {
                return `${name}[${Math.floor(index / 4)}][${index % 4}]`;
            }
        }
    }
    else {
        return length > 1 ? `${name}[${index}]` : name;
    }
};
/**
 * A helper function to get a IndicesHelper for a given input or output.
 *
 * @param name - the name of the input or output.
 * @param tensorType - the tensor type of the input or output.
 * @param shapeOrRank - the tensor shape or the rank of the input or output.
 * @param usage - the usage of the indices helper.
 * @param components - indicates the number of components of each element. 1 for scalar, 2 for vec2, 3 for vec3, 4 for
 *    vec4.
 */
const createIndicesHelper = (name, tensorType, shapeOrRank, usage, components) => {
    const useUniform = typeof shapeOrRank === 'number';
    const rank = useUniform ? shapeOrRank : shapeOrRank.length;
    const rankIdentity = [...new Array(rank).keys()];
    const indicesType = rank < 2 ? 'u32' : rank <= 4 ? `vec${rank}<u32>` : `array<u32, ${rank}>`;
    const mappedType = getWgslMappedType(tensorType, components);
    const valueType = typeof mappedType === 'string' ? mappedType : mappedType[1];
    const storageType = typeof mappedType === 'string' ? mappedType : mappedType[0];
    const type = { indices: indicesType, value: valueType, storage: storageType, tensor: tensorType };
    const normalizeDim = (dim) => typeof dim === 'string' ? dim : `${dim}u`;
    const implementationUsed = {
        offsetToIndices: false,
        indicesToOffset: false,
        broadcastedIndicesToOffset: false,
        set: false,
        setByIndices: false,
        get: false,
        getByIndices: false,
    };
    const uniformPrefix = useUniform ? 'uniforms.' : '';
    const shape = `${uniformPrefix}${name}_shape`;
    const strides = `${uniformPrefix}${name}_strides`;
    let o2iSnippet = '';
    for (let i = 0; i < rank - 1; i++) {
        o2iSnippet += `
    let dim${i} = current / ${getElementAt(strides, i, rank)};
    let rest${i} = current % ${getElementAt(strides, i, rank)};
    indices[${i}] = dim${i};
    current = rest${i};
    `;
    }
    o2iSnippet += `indices[${rank - 1}] = current;`;
    const offsetToIndicesImplementation = rank < 2 ? '' : `
  fn o2i_${name}(offset: u32) -> ${type.indices} {
    var indices: ${type.indices};
    var current = offset;
    ${o2iSnippet}
    return indices;
  }`;
    const offsetToIndices = (varOffset) => {
        implementationUsed.offsetToIndices = true;
        return rank < 2 ? varOffset : `o2i_${name}(${varOffset})`;
    };
    const offsets = [];
    if (rank >= 2) {
        for (let i = rank - 1; i >= 0; i--) {
            offsets.push(`${getElementAt(strides, i, rank)} * (indices[${i}])`);
        }
    }
    const indicesToOffsetImplementation = rank < 2 ? '' : `
  fn i2o_${name}(indices: ${type.indices}) -> u32 {
    return ${offsets.join('+')};
  }`;
    const indicesToOffset = (varIndices) => {
        implementationUsed.indicesToOffset = true;
        return rank < 2 ? varIndices : `i2o_${name}(${varIndices})`;
    };
    const indices = (...init) => rank === 0 ? '0u' : `${type.indices}(${init.map(normalizeDim).join(',')})`;
    const indicesGet = (varIndices, idx) => {
        if (rank < 2) {
            return `${varIndices}`;
        }
        else {
            return `${getElementAt(varIndices, idx, rank)}`;
        }
    };
    const indicesSet = (varIndices, idx, value) => {
        if (rank < 2) {
            return `${varIndices}=${value};`;
        }
        else {
            return `${getElementAt(varIndices, idx, rank)}=${value};`;
        }
    };
    const broadcastedIndicesToOffsetImplementation = {};
    const broadcastedIndicesToOffset = (varIndices, output) => {
        implementationUsed.broadcastedIndicesToOffset = true;
        const implKey = `${output.name}broadcastedIndicesTo${name}Offset`;
        if (implKey in broadcastedIndicesToOffsetImplementation) {
            return `${implKey}(${varIndices})`;
        }
        const offsets = [];
        for (let i = rank - 1; i >= 0; i--) {
            const idx = output.indicesGet('outputIndices', i + output.rank - rank);
            offsets.push(`${indicesGet(strides, i)} * (${idx} % ${indicesGet(shape, i)})`);
        }
        broadcastedIndicesToOffsetImplementation[implKey] =
            `fn ${implKey}(outputIndices: ${output.type.indices}) -> u32 {
             return ${offsets.length > 0 ? offsets.join('+') : '0u'};
           }`;
        return `${implKey}(${varIndices})`;
    };
    const setByOffset = (offset, value) => (() => {
        if (type.storage === type.value) {
            return `${name}[${offset}]=${value};`;
        }
        else if (type.storage === 'vec2<u32>' && type.value === 'i32') {
            // int64, components === 1
            return `${name}[${offset}]=vec2<u32>(u32(${value}), select(0u, 0xFFFFFFFFu, ${value} < 0));`;
        }
        else if (type.storage === 'vec2<u32>' && type.value === 'u32') {
            // uint64, components === 1
            return `${name}[${offset}]=vec2<u32>(u32(${value}), 0u);`;
        }
        else if (type.storage === 'u32' && type.value === 'vec4<bool>') {
            // bool, components === 4
            return `${name}[${offset}]=dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(${value}));`;
        }
        else {
            throw new Error(`not supported combination of storage type ${type.storage} and value type ${type.value} yet`);
        }
    })();
    const getByOffset = (offset) => (() => {
        if (type.storage === type.value) {
            return `${name}[${offset}]`;
        }
        else if (type.storage === 'vec2<u32>' && type.value === 'i32') {
            // int64, components === 1
            return `i32(${name}[${offset}].x)`;
        }
        else if (type.storage === 'vec2<u32>' && type.value === 'u32') {
            // uint64, components === 1
            return `u32(${name}[${offset}].x)`;
        }
        else if (type.storage === 'u32' && type.value === 'vec4<bool>') {
            // bool, components === 4
            return `vec4<bool>(bool(${name}[${offset}] & 0xFFu), bool(${name}[${offset}] & 0xFF00u), bool(${name}[${offset}] & 0xFF0000u), bool(${name}[${offset}] & 0xFF000000u))`;
        }
        else {
            throw new Error(`not supported combination of storage type ${type.storage} and value type ${type.value} yet`);
        }
    })();
    const getByIndicesImplementation = rank < 2 ? '' : `
  fn get_${name}ByIndices(indices: ${type.indices}) -> ${valueType} {
    return ${getByOffset(`i2o_${name}(indices)`)};
  }`;
    const getImplementation = rank < 2 ? '' : (() => {
        const functionParams = rankIdentity.map(i => `d${i}: u32`).join(', ');
        const dimsParams = rankIdentity.map(i => `d${i}`).join(', ');
        return `
  fn get_${name}(${functionParams}) -> ${valueType} {
    return get_${name}ByIndices(${indices(dimsParams)});
  }`;
    })();
    const get = (...indices) => {
        if (indices.length !== rank) {
            throw new Error(`indices length must be ${rank}`);
        }
        const normalizedIndices = indices.map(normalizeDim).join(',');
        if (rank === 0) {
            return getByOffset('0u');
        }
        else if (rank === 1) {
            return getByOffset(normalizedIndices[0]);
        }
        else {
            implementationUsed.get = true;
            implementationUsed.getByIndices = true;
            implementationUsed.indicesToOffset = true;
            return `get_${name}(${normalizedIndices})`;
        }
    };
    const getByIndices = (varIndices) => {
        if (rank < 2) {
            return getByOffset(varIndices);
        }
        else {
            implementationUsed.getByIndices = true;
            implementationUsed.indicesToOffset = true;
            return `get_${name}ByIndices(${varIndices})`;
        }
    };
    const setByIndicesImplementation = rank < 2 ? '' : `
  fn set_${name}ByIndices(indices: ${type.indices}, value: ${valueType}) {
    ${setByOffset(`i2o_${name}(indices)`, 'value')}
  }`;
    const setImplementation = rank < 2 ? '' : (() => {
        const functionParams = rankIdentity.map(i => `d${i}: u32`).join(', ');
        const dimsParams = rankIdentity.map(i => `d${i}`).join(', ');
        return `
  fn set_${name}(${functionParams}, value: ${valueType}) {
    set_${name}ByIndices(${indices(dimsParams)}, value);
  }`;
    })();
    const set = (...indicesAndValue) => {
        if (indicesAndValue.length !== rank + 1) {
            throw new Error(`indices length must be ${rank}`);
        }
        const value = indicesAndValue[rank];
        if (typeof value !== 'string') {
            throw new Error('value must be string');
        }
        const normalizedIndices = indicesAndValue.slice(0, rank).map(normalizeDim).join(',');
        if (rank === 0) {
            return setByOffset('0u', value);
        }
        else if (rank === 1) {
            return setByOffset(normalizedIndices[0], value);
        }
        else {
            implementationUsed.set = true;
            implementationUsed.setByIndices = true;
            implementationUsed.indicesToOffset = true;
            return `set_${name}(${normalizedIndices}, ${value})`;
        }
    };
    const setByIndices = (varIndices, value) => {
        if (rank < 2) {
            return setByOffset(varIndices, value);
        }
        else {
            implementationUsed.setByIndices = true;
            implementationUsed.indicesToOffset = true;
            return `set_${name}ByIndices(${varIndices}, ${value});`;
        }
    };
    const impl = () => {
        const impls = [];
        let needShapeStrides = false;
        if (implementationUsed.offsetToIndices) {
            impls.push(offsetToIndicesImplementation);
            needShapeStrides = true;
        }
        if (implementationUsed.indicesToOffset) {
            impls.push(indicesToOffsetImplementation);
            needShapeStrides = true;
        }
        if (implementationUsed.broadcastedIndicesToOffset) {
            Object.values(broadcastedIndicesToOffsetImplementation).forEach(impl => impls.push(impl));
            needShapeStrides = true;
        }
        if (implementationUsed.set) {
            impls.push(setImplementation);
            needShapeStrides = true;
        }
        if (implementationUsed.setByIndices) {
            impls.push(setByIndicesImplementation);
            needShapeStrides = true;
        }
        if (implementationUsed.get) {
            impls.push(getImplementation);
            needShapeStrides = true;
        }
        if (implementationUsed.getByIndices) {
            impls.push(getByIndicesImplementation);
            needShapeStrides = true;
        }
        if (!useUniform && needShapeStrides) {
            impls.unshift(`const ${shape} = ${type.indices}(${shapeOrRank.join(',')});`, `const ${strides} = ${type.indices}(${ShapeUtil.computeStrides(shapeOrRank).join(',')});`);
        }
        return impls.join('\n');
    };
    return {
        impl,
        type,
        offsetToIndices,
        indicesToOffset,
        broadcastedIndicesToOffset,
        indices,
        indicesGet,
        indicesSet,
        set,
        setByOffset,
        setByIndices,
        get,
        getByOffset,
        getByIndices,
        // isVec4,
        usage,
        name,
        strides,
        shape,
        rank
    };
};
/**
 * Create a IndicesHelper for an input.
 *
 * @param name - the name of the input.
 * @param type - the tensor type of the input.
 * @param shapeOrRank - the tensor shape or the rank of the input.
 * @param components - the number of components of the input. available values are 1, 2, 3, 4. default is 1.
 * @returns an IndicesHelper for the input.
 */
export const inputVariable = (name, type, shapeOrRank, components = 1) => createIndicesHelper(name, type, shapeOrRank, 'input', components);
/**
 * Create a IndicesHelper for an output.
 *
 * @param name - the name of the output.
 * @param type - the tensor type of the output.
 * @param shapeOrRank - the tensor shape or the rank of the output.
 * @param components - the number of components of the output. available values are 1, 2, 3, 4. default is 1.
 * @returns an IndicesHelper for the output.
 */
export const outputVariable = (name, type, shapeOrRank, components = 1) => createIndicesHelper(name, type, shapeOrRank, 'output', components);
/**
 * Create a IndicesHelper for an internal variable.
 *
 * @param name - the name of the variable.
 * @param type - the tensor type of the variable.
 * @param shapeOrRank - the tensor shape or the rank of the variable.
 * @param components - the number of components of the variable. available values are 1, 2, 3, 4. default is 1.
 * @returns an IndicesHelper for the variable.
 */
export const internalVariable = (name, type, shapeOrRank, components = 1) => createIndicesHelper(name, type, shapeOrRank, 'internal', components);
class ShaderHelperImpl {
    constructor(normalizedDispatchGroup) {
        this.normalizedDispatchGroup = normalizedDispatchGroup;
        this.internalVariables = [];
        this.variables = [];
        this.uniforms = [];
        this.variableIndex = 0;
    }
    guardAgainstOutOfBoundsWorkgroupSizes(size) {
        // Guard against out-of-bounds work group sizes
        const sizeInCode = typeof size === 'number' ? `${size}u` : size;
        return `if (global_idx >= ${sizeInCode}) { return; }`;
    }
    mainStart(workgroupSize = WORKGROUP_SIZE) {
        const workgroupSizeX = typeof workgroupSize === 'number' ? workgroupSize : workgroupSize[0];
        const workgroupSizeY = typeof workgroupSize === 'number' ? 1 : workgroupSize[1];
        const workgroupSizeZ = typeof workgroupSize === 'number' ? 1 : workgroupSize[2];
        const is1DimensionDispatch = this.normalizedDispatchGroup[1] === 1 && this.normalizedDispatchGroup[2] === 1;
        const paramList = is1DimensionDispatch ? `@builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>` :
            `@builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(local_invocation_index) local_idx : u32,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(num_workgroups) num_workgroups : vec3<u32>`;
        const globalIdxDefinition = is1DimensionDispatch ?
            'let global_idx = global_id.x; let local_idx = local_id.x;' :
            `let global_idx = (workgroup_id.z * num_workgroups[0] * num_workgroups[1] +
          workgroup_id.y * num_workgroups[0] + workgroup_id.x) * ${workgroupSizeX * workgroupSizeY * workgroupSizeZ}u + local_idx;`;
        return `@compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY}, ${workgroupSizeZ})
  fn main(${paramList}) {
    ${globalIdxDefinition}
  `;
    }
    appendVariableUniforms(variable) {
        if (variable.rank !== 0) {
            if (variable.shape.startsWith('uniforms.')) {
                this.uniforms.push({ name: variable.shape.replace('uniforms.', ''), type: 'u32', length: variable.rank });
            }
            if (variable.strides.startsWith('uniforms.')) {
                this.uniforms.push({ name: variable.strides.replace('uniforms.', ''), type: 'u32', length: variable.rank });
            }
        }
    }
    declareVariable(variable, bindingIndex) {
        if (variable.usage === 'internal') {
            throw new Error('cannot use internal variable with declareVariable(). use registerInternalVariables() instead.');
        }
        this.variables.push(variable);
        this.appendVariableUniforms(variable);
        const access = variable.usage === 'input' ? 'read' : 'read_write';
        const storageType = variable.type.storage;
        return `@group(0) @binding(${bindingIndex}) var<storage, ${access}> ${variable.name}: array<${storageType}>;`;
    }
    declareVariables(...variables) {
        return variables.map(v => this.declareVariable(v, this.variableIndex++)).join('\n');
    }
    registerInternalVariable(variable) {
        if (variable.usage !== 'internal') {
            throw new Error('cannot use input or output variable with registerInternalVariable(). use declareVariables() instead.');
        }
        this.internalVariables.push(variable);
        this.appendVariableUniforms(variable);
    }
    registerInternalVariables(...variables) {
        variables.forEach(v => this.registerInternalVariable(v));
        return this;
    }
    registerUniform(name, type, length = 1) {
        this.uniforms.push({ name, type, length });
        return this;
    }
    registerUniforms(additionalUniforms) {
        this.uniforms = this.uniforms.concat(additionalUniforms);
        return this;
    }
    uniformDeclaration() {
        if (this.uniforms.length === 0) {
            return '';
        }
        const uniformSnippets = [];
        for (const { name, type, length } of this.uniforms) {
            if (length && length > 4) {
                if (type === 'f16') {
                    uniformSnippets.push(`@align(16) ${name}:array<mat2x4<${type}>, ${Math.ceil(length / 8)}>`);
                }
                else {
                    uniformSnippets.push(`${name}:array<vec4<${type}>, ${Math.ceil(length / 4)}>`);
                }
            }
            else {
                const typeTemp = length == null || length === 1 ? type : `vec${length}<${type}>`;
                uniformSnippets.push(`${name}:${typeTemp}`);
            }
        }
        return `
      struct Uniforms { ${uniformSnippets.join(', ')} };
      @group(0) @binding(${this.variableIndex}) var<uniform> uniforms: Uniforms;`;
    }
    /**
     * Get additional implementation that needs to be added to the shader source.
     */
    get additionalImplementations() {
        return this.uniformDeclaration() + this.variables.map(i => i.impl()).join('\n') +
            this.internalVariables.map(i => i.impl()).join('\n');
    }
}
export const createShaderHelper = (dispatchGroup) => new ShaderHelperImpl(dispatchGroup);
/**
 * This function comes from https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/ops/broadcast_util.ts#L18-L40
 * Returns the dimensions in the input shape that are broadcasted to
 * produce the provided output shape.
 *
 * The returned dimensions are 0-indexed and sorted. An example:
 * inShape = [4, 1, 3]
 * outShape = [5, 4, 3, 3]
 * result = [1]. Dimension 1 (2nd dimension of input) gets broadcasted 1 => 3.
 */
export const getBroadcastDims = (inShape, outShape) => {
    const inRank = inShape.length;
    const dims = [];
    for (let i = 0; i < inRank; i++) {
        const dim = inRank - 1 - i;
        const a = inShape[dim] || 1;
        const b = outShape[outShape.length - 1 - i] || 1;
        if (b > 1 && a === 1) {
            dims.unshift(dim);
        }
    }
    return dims;
};
