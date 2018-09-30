#version 400

uniform sampler2D tex0;

in vec2 Vertex_UV;
in vec4 int_Color;

out vec4 out_Color;

void main(void)
{
  //vec2 uv = Vertex_UV.xy;
  //uv.y *= -1.0;
  vec4 basecolor = int_Color;//texture(tex0,uv);
  //if(basecolor.a <= 0.1){
  //    discard;
  //}
  out_Color = basecolor;
  //out_Color.x *= int_Color.x;
  //out_Color.y *= int_Color.y;
  //out_Color.z *= int_Color.z;
  //out_Color.w *= int_Color.w;
  //out_Color = int_Color;
}