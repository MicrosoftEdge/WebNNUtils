class CSRunner {
    constructor(device, shader_text, name) {
        this.device = device;
        this.shader_text = shader_text;
        this.name = name;
    }

    Initialize()
    {
        this.module = this.device.createShaderModule({
            label: 'Compute shader ' + this.name,
            code:this.shader_text
        });

        this.pipelineCompute = this.device.createComputePipeline({
            label: 'Compute pipeline ' + this.name,
            layout: 'auto',
            compute: {
                module: this.module,
                entryPoint: 'main',
            },
        });
    }

    GenerateComputePass(encoder, bindGroupEntries, channel, width, height)
    {             
       const bindGroupCS = this.device.createBindGroup({
        label: 'Compute bindgroup ' + this.name,
        layout: this.pipelineCompute.getBindGroupLayout(0),
        entries: bindGroupEntries,
       });
       const computePass = encoder.beginComputePass({
            label: 'Compute pass ' + this.name,
        });
        computePass.setPipeline(this.pipelineCompute);
        computePass.setBindGroup(0, bindGroupCS);
        computePass.dispatchWorkgroups(Math.ceil(width / 8.0), Math.ceil(height / 8.0), channel);
        computePass.end();
    } 
}