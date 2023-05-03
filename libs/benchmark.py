from timeit import default_timer as timer


def benchmark(test_loops, pipe, prompt, width, height, generator, num_inference_steps, guidance_scale, output):
    durtions = []
    for i in range(test_loops):
      start = timer()
      image = pipe(prompt,
                  width=width,
                  height=height,
                  generator=generator,
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                  ).images[0]
      end = timer()
      image.save(f"output/{output}-{i}.png")
      durtion = round(end - start, 3)
      print(f"test {i}: {durtion}s")
    durtions.append(durtion)
    mean_durtion = round(sum(durtions)/len(durtions),3)
    print(f"mean duration: {mean_durtion}s")
