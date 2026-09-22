// Gmsh project
// Elastic half space dimensions
p0x = 0.0; p0y = 0.0;
p1x = 2.2; p1y = 2.0;
lc = 0.03;

// Third medium dimensions
p2y = 4.0;

// Cylinder dimensions
R = 1.0;

// Create points, lines, circle arcs
Point(1) = {p0x, p0y, 0, lc};
Point(2) = {p0x, p1y, 0, lc};
Point(3) = {p1x, p1y, 0, lc};
Point(4) = {p1x, p0y, 0, lc};
Point(5) = {p0x, p2y, 0, lc};
Point(6) = {p1x, p2y, 0, lc};
Point(7) = {p1x/2, p1y+R+0.05, 0, lc};
Point(8) = {1.1-Sqrt(0.0975), 4, 0, lc};
Point(9) = {1.1+Sqrt(0.0975), 4, 0, lc};
Point(10) = {p1x/2, p1y+0.05, 0, lc};
Point(11) = {p1x/2, p1y, 0, lc};
Point(12) = {0.85, 4, 0, lc};
Point(13) = {1.35, 4, 0, lc};
Point(14) = {p1x/2-0.12, p1y, 0, 1*lc};
Point(15) = {p1x/2+0.12, p1y, 0, 1*lc};

Line(1) = {1, 2};
Line(2) = {2, 14};
Line(15) = {14, 11};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 2};
Line(6) = {11, 15};
Line(16) = {15, 3};
Line(7) = {3, 6};
Circle(8) = {8,7,10};
Circle(9) = {10,7,9};
Line(10) = {9, 13};
Line(13) = {13, 12};
Line(14) = {12, 8};
Line(11) = {8, 5};
Line(12) = {6, 9};

// Create surface
Curve Loop(1) = {1, 2, 15, 6, 16, 3, 4}; // elastic half space
Plane Surface(1) = {1};
Curve Loop(2) = {10, 13, 14, 8, 9};      // inner hole (cylinder)
Plane Surface(2) = {2}; 
Curve Loop(3) = {5, 2, 15, 6, 16, 7, 12, 10, 13, 14, 11};
Plane Surface(3) = {3, 2};               // third medium

// Physical Groups for materials
Physical Surface("elastic_half_space", 1) = {1};  // Material tag 1, dx(1)
Physical Surface("contact_body", 2) = {2};        // Material tag 2, dx(2)  
Physical Surface("third_medium", 3) = {3};        // Material tag 3, dx(3)

// Physical Groups for boundaries
Physical Curve("bottom_boundary", 10) = {4};      // Bottom edge, ds(10)
Physical Curve("top_boundary", 11) = {13};        // Top edge, ds(11)

// Circular refined region with radius
Field[1] = Ball;
Field[1].VIn = 0.02*lc;
Field[1].VOut = lc;
Field[1].XCenter = 1.1;
Field[1].YCenter = 2.025; 
Field[1].Radius = 0.1;
Field[1].Thickness = 0.2;
Background Field = 1;

// Mesh control
Mesh.ElementOrder = 1;        // first-order elements
Mesh.Algorithm = 8;
Mesh 2;