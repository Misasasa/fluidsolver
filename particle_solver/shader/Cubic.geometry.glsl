#version 400

layout (points) in;
layout (triangle_strip,max_vertices=24) out;

uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

uniform float particle_size;

in VS_OUT{
  vec4 ex_Color;
  vec4 exvec0;
  vec4 exvec1;
  vec4 exvec2;
  vec4 exvec3;
  } gs_in[];

out vec2 Vertex_UV;
out vec4 int_Color;

void generateCube(vec4 P, mat4 localRotation){

  mat4 projview = ProjectionMatrix * ViewMatrix;
  //up
  gl_Position = P;
  //int_Color=vec4(1,0,0,1);
  //EmitVertex();
  //int_Color =  gs_in[0].ex_Color;
  
  int_Color = vec4(0.6,0.2,0.2,1);
  vec3 displacement =  vec3(-0.5,0.5,0.5) * particle_size;
  vec4 rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(-0.5,0.5,-0.5) * particle_size;
  //rotatedDx=LocalRotation[0]*vec4(displacement,1);
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(0.5,0.5,0.5) * particle_size;
  //rotatedDx=LocalRotation[0]*vec4(displacement,1);
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(0.5,0.5,-0.5) * particle_size;
  //rotatedDx=LocalRotation[0]*vec4(displacement,1);
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);

  EmitVertex();

  EndPrimitive();



  //down
  int_Color = vec4(0.2,0.6,0.2,1);
  displacement =  vec3(-0.5,-0.5,0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(-0.5,-0.5,-0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(0.5,-0.5,0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(0.5,-0.5,-0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  EndPrimitive();





  //left
  int_Color = vec4(0.2,0.2,0.6,1);
  displacement =  vec3(-0.5,-0.5,-0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(-0.5,0.5,-0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(-0.5,-0.5,0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(-0.5,0.5,0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  EndPrimitive();



  //right
  int_Color = vec4(0.2,0.4,0.6,1);
  displacement =  vec3(0.5,-0.5,-0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(0.5,0.5,-0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(0.5,-0.5,0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(0.5,0.5,0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  EndPrimitive();




  //front
  int_Color = vec4(0.6,0.4,0.2,1);
  displacement =  vec3(-0.5,-0.5,0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(-0.5,0.5,0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(0.5,-0.5,0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(0.5,0.5,0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  EndPrimitive();





  //back
  int_Color = vec4(0.1,0.6,0.6,1);
  displacement =  vec3(-0.5,-0.5,-0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(-0.5,0.5,-0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(0.5,-0.5,-0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  displacement = vec3(0.5,0.5,-0.5) * particle_size;
  rotatedDx = localRotation * vec4(displacement, 0); 
  //gl_Position = projview * (P + vec4(displacement,0));
  gl_Position = projview * (P + rotatedDx);
  EmitVertex();

  EndPrimitive();



}

void main(void)
{

  vec4 P = gl_in[0].gl_Position;

  P = P;
  //
  mat4 ex_Rotation;
  ex_Rotation[0][0] = gs_in[0].exvec0.x; ex_Rotation[1][0] = gs_in[0].exvec0.y; ex_Rotation[2][0] = gs_in[0].exvec0.z; ex_Rotation[3][0] = gs_in[0].exvec0.w;
  ex_Rotation[0][1] = gs_in[0].exvec1.x; ex_Rotation[1][1] = gs_in[0].exvec1.y; ex_Rotation[2][1] = gs_in[0].exvec1.z; ex_Rotation[3][1] = gs_in[0].exvec1.w;
  ex_Rotation[0][2] = gs_in[0].exvec2.x; ex_Rotation[1][2] = gs_in[0].exvec2.y; ex_Rotation[2][2] = gs_in[0].exvec2.z; ex_Rotation[3][2] = gs_in[0].exvec2.w;
  ex_Rotation[0][3] = gs_in[0].exvec3.x; ex_Rotation[1][3] = gs_in[0].exvec3.y; ex_Rotation[2][3] = gs_in[0].exvec3.z; ex_Rotation[3][3] = gs_in[0].exvec3.w;


  // a: left-bottom

  //int_Color = gs_in[0].ex_Color;
  //vec4 tmp = ex_Rotation*vec4(0,1,0,1);
  //tmp.x = abs(tmp.x);
  //tmp.y = abs(tmp.y);
  //tmp.z = abs(tmp.z);
  //int_Color = tmp;

  generateCube(P, ex_Rotation);
  //vec2 va = P.xy + vec2(-0.5,-0.5)*particle_size;
  //gl_Position = ProjectionMatrix  * vec4(va,P.zw);
  //Vertex_UV = vec2(0.0, 0.0);
  
  //EmitVertex();

  // b: left-top
  //vec2 vb = P.xy + vec2(-0.5,0.5)*particle_size;
  //gl_Position = ProjectionMatrix *  vec4(vb,P.zw);
  //Vertex_UV = vec2(0.0, 1.0);
  //EmitVertex();

  // c: right-bottom
  //vec2 vd = P.xy + vec2(0.5,-0.5)*particle_size;
  //gl_Position = ProjectionMatrix * vec4(vd,P.zw);
  //Vertex_UV = vec2(1.0, 0.0);
  //EmitVertex();

  // d: right-top
  //vec2 vc = P.xy + vec2(0.5,0.5)*particle_size;
  //gl_Position = ProjectionMatrix * vec4(vc,P.zw);
  //Vertex_UV = vec2(1.0, 1.0);
  //EmitVertex();
  

  //EndPrimitive();
}